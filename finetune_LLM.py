import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

save_dir = "./ppo_checkpoints"
os.makedirs(save_dir, exist_ok=True)

save_every = 10  # 每幾步儲存一次模型

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"
REWARD_MODEL_NAME = "roberta-base"   # 可改成 roberta-large (4090 扛得住)
USE_FP16 = True
BATCH_SIZE = 8
EPOCHS = 5
MAX_NEW_TOKENS = 400

# --- LOAD TOKENIZER AND BASE MODEL ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)


# --- LoRA CONFIGURATION ---

# "mlp.w1",
# "mlp.w2",
# "mlp.c_proj"

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.c_attn", "attn.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
model.eval()

# --- Reward Model ---
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward-style-model").to("cuda")

def get_reward(texts):
    inputs = reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = reward_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    return probs[:, 1].detach().cpu().tolist()


from copy import deepcopy
from torch.optim import Adam

# 拷貝舊策略
old_model = deepcopy(model).eval()
optimizer = Adam(model.parameters(), lr=1e-5)
eps_clip = 0.2

# 範例 prompts
prompts = [
    "請你幫我寫一句溫和但堅定的拒絕句子。",
    "寫一段開場白，讓人感受到理性與禮貌。",
    "如何有禮貌地指出對方的錯誤？"
]

# 開始 PPO 訓練
for epoch in range(EPOCHS):
    for step, prompt in enumerate(prompts):
        model.train()

        # 1. 生成輸出句子
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            response_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        # 2. 合併 prompt + response
        full_input = torch.cat([input_ids, response_ids[:, input_ids.shape[-1]:]], dim=1)
        attention_mask = full_input.ne(tokenizer.pad_token_id).long()
        decoded_text = tokenizer.decode(full_input[0], skip_special_tokens=True)

        # 3. reward 計算與 baseline
        reward = get_reward([decoded_text])[0]
        baseline = 0.5
        advantage = reward - baseline

        # 4. 策略 log_prob（新模型）
        labels = full_input.clone()
        log_probs = model(input_ids=full_input, attention_mask=attention_mask, labels=labels).logits
        log_probs = F.log_softmax(log_probs, dim=-1)
        chosen_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        log_prob_sum = chosen_log_probs.sum(dim=-1)

        # 5. 舊策略 log_prob
        with torch.no_grad():
            old_logits = old_model(input_ids=full_input, attention_mask=attention_mask, labels=labels).logits
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_chosen_log_probs = torch.gather(old_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            old_log_prob_sum = old_chosen_log_probs.sum(dim=-1)

        # 6. 計算 PPO loss
        ratio = torch.exp(log_prob_sum - old_log_prob_sum)
        clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
        ppo_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        # 7. 更新
        optimizer.zero_grad()
        ppo_loss.backward()
        optimizer.step()

        # 8. 同步 old policy
        old_model.load_state_dict(model.state_dict())

        print(f"[Epoch {epoch}] Step {step} | Reward: {reward:.4f} | Loss: {ppo_loss.item():.4f}")
        print(f"→ Output: {decoded_text}\n")

        # log to TensorBoard
        global_step = epoch * len(prompts) + step
        writer.add_scalar("Reward", reward, global_step)
        writer.add_scalar("PPO_Loss", ppo_loss.item(), global_step)

        # save model checkpoint
        if global_step % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_step{global_step}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"✅ Saved model to: {checkpoint_path}")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from collections import deque
import numpy as np
import os

# === TensorBoard and Checkpoint Directories ===
log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

save_dir = "./ppo_checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_every = 100

# === Configuration ===
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"
REWARD_MODEL_NAME = "roberta-base"
EPOCHS = 20
MAX_NEW_TOKENS = 400
LR = 1e-5
EPS_CLIP = 0.2

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.eos_token = tokenizer.eos_token or "<|endoftext|>"
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
assert tokenizer.pad_token_id is not None, "❌ pad_token_id 還是 None"

# === LoRA Configuration ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.c_attn", "attn.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# === Model and Old Model Initialization ===
def init_peft_model():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return get_peft_model(base, lora_config).eval()

model = init_peft_model()
old_model = init_peft_model()
old_model.load_state_dict(model.state_dict())
optimizer = Adam(model.parameters(), lr=LR)

# === Reward Model ===
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward-style-model").to("cuda")

def get_reward(texts):
    inputs = reward_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = reward_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    return probs[:, 1].detach().cpu().tolist()

import json
# === Training Prompts ===
with open("ml_prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)

from tqdm import tqdm
from collections import deque
import numpy as np

reward_window = deque(maxlen=50)  # 平滑 reward 用
best_avg_reward = -float("inf")
last_checkpoint_dir = os.path.join(save_dir, "last_checkpoint")
best_checkpoint_dir = os.path.join(save_dir, "best_checkpoint")

for epoch in range(EPOCHS):
    epoch_rewards = []
    epoch_losses = []
    epoch_kls = []

    loop = tqdm(prompts, desc=f"Epoch {epoch}", leave=False)
    skipped_steps = 0

    for step, prompt in enumerate(loop):
        model.train()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            input_attention_mask = input_ids.ne(tokenizer.pad_token_id)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

        response_ids = generated[:, input_ids.shape[-1]:]
        attention_mask = response_ids.ne(tokenizer.pad_token_id).long()
        full_input = torch.cat([input_ids, response_ids], dim=1)
        decoded_text = tokenizer.decode(full_input[0], skip_special_tokens=True)

        reward = get_reward([decoded_text])[0]
        reward = np.clip(reward, -1.0, 2.0)  # ✅ reward clipping
        reward_window.append(reward)
        avg_reward = np.mean(reward_window)
        baseline = avg_reward
        advantage = reward - baseline
        advantage = float(np.clip(advantage, -5.0, 5.0))  # ✅ advantage clipping

        labels = response_ids.clone()
        logits = model(input_ids=response_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        log_prob_sum = chosen_log_probs.sum(dim=-1)

        with torch.no_grad():
            old_logits = old_model(input_ids=response_ids, attention_mask=attention_mask).logits
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_chosen_log_probs = torch.gather(old_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            old_log_prob_sum = old_chosen_log_probs.sum(dim=-1)

        ratio = torch.exp(log_prob_sum - old_log_prob_sum)
        clipped_ratio = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)

        # ✅ Optional: KL penalty
        kl_div = (old_log_prob_sum - log_prob_sum).mean()
        kl_penalty_weight = 0.1  # 可調整的權重
        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        policy_loss += kl_penalty_weight * kl_div  # ✅ 加入 KL 項（鼓勵與舊策略靠近）

        MAX_LOSS = 10.0
        MAX_KL = 2.0
        MIN_KL = 0.0

        if not torch.isfinite(policy_loss) or abs(policy_loss.item()) > MAX_LOSS:
            skipped_steps += 1
            continue
        if not np.isfinite(kl_div.item()) or kl_div.item() < MIN_KL or kl_div.item() > MAX_KL:
            skipped_steps += 1
            continue
        if not torch.isfinite(ratio).all():
            skipped_steps += 1
            continue
        if not torch.isfinite(log_prob_sum).all() or not torch.isfinite(old_log_prob_sum).all():
            skipped_steps += 1
            continue

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ gradient clipping
        optimizer.step()
        old_model.load_state_dict(model.state_dict())

        global_step = epoch * len(prompts) + step
        writer.add_scalar("Reward/Instant", reward, global_step)
        writer.add_scalar("Reward/Avg", avg_reward, global_step)
        writer.add_scalar("Advantage", advantage, global_step)
        writer.add_scalar("Loss/Policy_Loss", policy_loss.item(), global_step)
        writer.add_scalar("KL_Divergence", kl_div.item(), global_step)

        epoch_rewards.append(reward)
        epoch_losses.append(policy_loss.item())
        epoch_kls.append(kl_div.item())

        if global_step % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_step{global_step}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"✅ Saved model to: {checkpoint_path}")

    print(f"⚠️ Epoch {epoch} skipped steps: {skipped_steps} / {len(prompts)}")
    print(
        f"📘 [Epoch {epoch}] AvgReward: {np.mean(epoch_rewards):.4f} | "
        f"AvgLoss: {np.mean(epoch_losses):.4f} | AvgKL: {np.mean(epoch_kls):.4f}"
    )

    model.save_pretrained(last_checkpoint_dir)
    tokenizer.save_pretrained(last_checkpoint_dir)

    if np.mean(epoch_rewards) > best_avg_reward:
        best_avg_reward = np.mean(epoch_rewards)
        model.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)
        print(f"🌟 New best model saved with AvgReward = {best_avg_reward:.4f}")

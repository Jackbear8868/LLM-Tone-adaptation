import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# --- CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"
REWARD_MODEL_NAME = "roberta-base"   # 可改成 roberta-large (4090 扛得住)
USE_FP16 = True
BATCH_SIZE = 8
EPOCHS = 5
MAX_NEW_TOKENS = 64

# --- LOAD TOKENIZER AND BASE MODEL ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)

for name, module in base_model.named_modules():
    if "proj" in name or "attn" in name:
        print(name)

# --- LoRA CONFIGURATION ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
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

# --- PPO CONFIGURATION ---
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    mini_batch_size=4,
    log_with="tensorboard",
    use_score_scaling=True,
    use_score_norm=True,
    early_stopping=True,
    target_kl=0.1
)
ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

# --- PROMPTS ---
prompts = [
    "請解釋牛頓第三定律", 
    "什麼是熵？",
    "AI 模型怎麼學習？", 
    "老師如何講解電場與磁場的關係？"
] * (BATCH_SIZE // 4)  # 調整長度

# --- TRAINING LOOP ---

tokenized_prompts = [
    tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) for prompt in prompts
]

def generate_responses(tokenized_prompts):
    responses = []
    for prompt_ids in tokenized_prompts:
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            output = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append(decoded)
    return responses


def tokenize_responses(responses):
    return [
        tokenizer(r, return_tensors="pt").input_ids[0].to(model.device)
        for r in responses
    ]

def log_outputs(prompts, responses, rewards, epoch):
    print(f"\n=== Epoch {epoch+1} ===")
    for i in range(len(prompts)):
        print(f"[{i+1}] Prompt: {prompts[i]}")
        print(f"     Response: {responses[i]}")
        print(f"     Reward: {rewards[i]:.4f}\n")

for epoch in range(EPOCHS):
    responses = generate_responses(tokenized_prompts)
    rewards = get_reward(responses)
    query_tensors = [ids[0] for ids in tokenized_prompts]
    response_tensors = tokenize_responses(responses)
    
    ppo_trainer.step(query_tensors, response_tensors, rewards)
    log_outputs(prompts, responses, rewards, epoch)


print("✅ 訓練完成")

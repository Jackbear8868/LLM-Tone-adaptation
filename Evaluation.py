import json
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForCausalLM,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==== TODO 區域 ==== #

# 模型與 tokenizer 設定
GENERATOR_MODEL_NAME = "./ppo_checkpoints/checkpoint_step1000"  # 自己訓練的 LLM 模型（需支援 chat_template）
# GENERATOR_MODEL_NAME = "./ppo_checkpoints/best_checkpoint"  # 自己訓練的 LLM 模型（需支援 chat_template）
BERT_MODEL_NAME = "./reward-style-model"         # 自己訓練的 RoBERTa 模型

# 超參數設定
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
MAX_INPUT_LENGTH = 1024  # 控制輸入長度，避免超過模型限制

# 檔案名稱
PROMPT_FILE = "prompts.json"
OUTPUT_FILE = "scored_generated_dialogues.json"
LEE_FILE = "label1.jsonl"
DATAS_FILE = "label0.jsonl"

# 向量檔案
LEE_VECTOR_FILE = "lee_vector.npy"
DATAS_VECTOR_FILE = "datas_vector.npy"
PURE_TONE_VECTOR_FILE = "pure_tone_vector.npy"

# =================== #

# 裝置設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# 模型載入
generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME, trust_remote_code=True)
generator_model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME, trust_remote_code=True).to(device)

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model.eval()

# 建立 LLM 生成器
generator = pipeline("text-generation", model=generator_model, tokenizer=generator_tokenizer,
                     device=0 if torch.cuda.is_available() else -1)

# 計算句子嵌入
def get_sentence_embedding(sentence):
    inputs = bert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        masked_hidden = last_hidden * mask
        summed = masked_hidden.sum(dim=1)
        count = mask.sum(dim=1)
        embedding = summed / count
        return embedding.squeeze().cpu().numpy()

# 建立語氣向量
def build_tone_vector(sentences, label=""):
    embeddings = [get_sentence_embedding(s) for s in tqdm(sentences, desc=f"建立向量: {label}")]
    return np.mean(embeddings, axis=0)

# 計算相似度
def score_against_vector(sentence, tone_vector):
    vec = get_sentence_embedding(sentence)
    return cosine_similarity([vec], [tone_vector])[0][0]

# 讀入向量或建立
def load_or_create_vector(vector_path, data_path, label):
    if os.path.exists(vector_path):
        choice = input(f"{label} 向量檔案已存在，要重新建立嗎？(y/n): ").strip().lower()
        if choice != "y":
            print(f"載入現有 {label} 向量...")
            return np.load(vector_path)

    with open(data_path, "r", encoding="utf-8") as f:
        sentences = [json.loads(line)['text'] for line in f]

    vec = build_tone_vector(sentences, label)
    np.save(vector_path, vec)
    print(f"{label} 向量已儲存至 {vector_path}")
    return vec

# 主流程：生成句子 + 評估語氣
def generate_and_evaluate():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    lee_vector = load_or_create_vector(LEE_VECTOR_FILE, LEE_FILE, "李宏毅")
    datas_vector = load_or_create_vector(DATAS_VECTOR_FILE, DATAS_FILE, "主題語意")
    pure_tone_vector = lee_vector - datas_vector
    np.save(PURE_TONE_VECTOR_FILE, pure_tone_vector)

    results = []
    scores_lee = []
    scores_datas = []
    scores_pure = []

    for i, prompt in enumerate(prompts):
        formatted_prompt = f"你是李宏毅老師，講話有熱情、詼諧且富有啟發性。\n\n使用者：{prompt}\n老師："

        input_ids = generator_tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=generator_tokenizer.model_max_length - MAX_NEW_TOKENS - 1).to(device)

        generation_output = generator_model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P
        )

        full_text = generator_tokenizer.decode(generation_output[0], skip_special_tokens=True)
        generated_text = full_text[len(formatted_prompt):].strip()

        score_lee = score_against_vector(generated_text, lee_vector)
        score_datas = score_against_vector(generated_text, datas_vector)
        score_pure = score_against_vector(generated_text, pure_tone_vector)

        print(f"[{i+1}] Prompt: {prompt}")
        print(f"==> 生成句子: {generated_text}")
        print(f" - 語氣相似 (Lee):      {score_lee:.3f}")
        print(f" - 主題相似 (Datas):   {score_datas:.3f}")
        print(f" - 純語氣 (Lee - Datas): {score_pure:.3f}")
        print("-" * 40)

        scores_lee.append(score_lee)
        scores_datas.append(score_datas)
        scores_pure.append(score_pure)

        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "score_lee": float(round(score_lee, 3)),
            "score_datas": float(round(score_datas, 3)),
            "score_pure_tone": float(round(score_pure, 3))
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    def print_stats(label, scores):
        avg = np.mean(scores)
        std = np.std(scores)
        print(f"[{label}] 平均: {avg:.3f}, 標準差: {std:.3f}")

    print("\n=== 統計結果 ===")
    print_stats("語氣相似 (Lee)", scores_lee)
    print_stats("主題相似 (Datas)", scores_datas)
    print_stats("純語氣 (Lee - Datas)", scores_pure)

if __name__ == "__main__":
    generate_and_evaluate()

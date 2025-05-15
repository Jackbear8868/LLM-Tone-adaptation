# ppl_evaluator.py

import json
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==== 設定區 ====
MODEL_NAME = "./ppo_checkpoints/checkpoint_step300"  # 可改成你 fine-tuned 的模型路徑
LEE_FILE = "label1.jsonl"            # 李宏毅語料資料集
DATAS_FILE = "label0.jsonl"       # 非李宏毅語料資料集
MAX_LEN = 512                    # 最長輸入長度（tokens）
USE_CUDA = torch.cuda.is_available()

# ==== 模型載入 ====
device = torch.device("cuda" if USE_CUDA else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

# ==== Perplexity 計算函數 ====
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return math.exp(loss.item())

# ==== 資料載入 ====
def load_sentences(path):
    sentences = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    sentences.append(item.get("text", "").strip())
                elif isinstance(item, str):
                    sentences.append(item.strip())
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sentences.append(obj.get("text", "").strip())
    return [s for s in sentences if s]

# ==== 評估流程 ====
def evaluate_ppl_on_dataset(label, file_path):
    print(f"\n🔍 開始評估 {label} 語料的 Perplexity ...")
    sentences = load_sentences(file_path)
    scores = []

    for sentence in tqdm(sentences, desc=f"{label}"):
        try:
            ppl = compute_perplexity(sentence)
            scores.append(ppl)
        except Exception as e:
            print(f"跳過失敗句子：{sentence[:30]}... -> {e}")
    
    scores = np.array(scores)
    print(f"\n📊 [{label}] 統計結果：")
    print(f" - 數量：{len(scores)}")
    print(f" - 平均 Perplexity：{scores.mean():.2f}")
    print(f" - 標準差：{scores.std():.2f}")

    return scores

# ==== 主程式 ====
if __name__ == "__main__":
    scores_lee = evaluate_ppl_on_dataset("李宏毅", LEE_FILE)
    scores_datas = evaluate_ppl_on_dataset("其他資料", DATAS_FILE)

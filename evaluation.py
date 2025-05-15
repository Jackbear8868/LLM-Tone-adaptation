import json, os, gc, torch, numpy as np
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForCausalLM, pipeline)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ================= 基本參數 ================= #
CHECKPOINT_DIR      = "./ppo_checkpoints"
CHECKPOINT_STEPS    = [800]      # 0,100,200,…,1000
BERT_MODEL_NAME     = "./best-model"
PROMPT_FILE         = "prompts2.json"
OUT_TEMPLATE        = "scored_generated_dialogues_step{}.json"

MAX_NEW_TOKENS      = 400
TEMPERATURE         = 0.7
TOP_K, TOP_P        = 50, 0.95
device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

# ================ 共用：tokenizer / BERT ================ #
# 假設所有 checkpoint 共用同一 tokenizer；以 step0 為基準載一次即可
BASE_TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{CHECKPOINT_STEPS[0]}")
generator_tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH, trust_remote_code=True)

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model     = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model.eval()

# ================ 嵌入相關函式 ================ #
@torch.no_grad()
def get_sentence_embedding(sentence:str):
    inputs = bert_tokenizer(sentence, return_tensors="pt",
                            truncation=True, max_length=128).to(device)
    outputs      = bert_model(**inputs).last_hidden_state
    mask         = inputs["attention_mask"].unsqueeze(-1)
    summed       = (outputs * mask).sum(dim=1)
    counts       = mask.sum(dim=1)
    return (summed / counts).squeeze().cpu().numpy()

def build_tone_vector(sentences, label=""):
    return np.mean([get_sentence_embedding(s) for s in tqdm(sentences, desc=f"建立向量:{label}")], axis=0)

def score_against_vector(sentence, tone_vec):
    return cosine_similarity([get_sentence_embedding(sentence)], [tone_vec])[0, 0]

def load_or_create_vector(vec_path:str, data_path:str, label:str):
    if os.path.exists(vec_path):
        return np.load(vec_path)
    with open(data_path, encoding="utf-8") as f:
        sents = [json.loads(l)["text"] for l in f]
    vec = build_tone_vector(sents, label)
    np.save(vec_path, vec)
    return vec
# ======== 載入 tone vectors (僅做一次) ======== #
LEE_VECTOR_FILE, DATAS_VECTOR_FILE, PURE_TONE_VECTOR_FILE = "lee_vector.npy", "datas_vector.npy", "pure_tone_vector.npy"
lee_vec   = load_or_create_vector(LEE_VECTOR_FILE,   "label1.jsonl", "李宏毅")
datas_vec = load_or_create_vector(DATAS_VECTOR_FILE, "label0.jsonl", "主題語意")
pure_vec  = lee_vec - datas_vec; np.save(PURE_TONE_VECTOR_FILE, pure_vec)

# ================ 讀取 prompts (一次即可) ================ #
with open(PROMPT_FILE, encoding="utf-8") as f:
    PROMPTS = json.load(f)

# =============== 統計用容器 (跨 checkpoint) =============== #
all_ckpt_avg = {"lee":[], "datas":[], "pure": []}

def print_step_stats(step, lee_s, datas_s, pure_s):
    def stat(arr): return np.mean(arr), np.std(arr)
    lee_m, lee_std     = stat(lee_s)
    datas_m, datas_std = stat(datas_s)
    pure_m, pure_std   = stat(pure_s)
    print(f"[Step {step}] Lee 平均/STD  : {lee_m:.3f} / {lee_std:.3f}")
    print( f"            Datas 平均/STD: {datas_m:.3f} / {datas_std:.3f}")
    print( f"            Pure  平均/STD: {pure_m:.3f} / {pure_std:.3f}\n")
    # 紀錄跨 checkpoint 的平均
    all_ckpt_avg["lee"].append(lee_m)
    all_ckpt_avg["datas"].append(datas_m)
    all_ckpt_avg["pure"].append(pure_m)

from tqdm import tqdm  # 確保這行在最上方有加

# =================== 主迴圈 =================== #
for step in tqdm(CHECKPOINT_STEPS, desc="Running checkpoints"):
    ckpt_path   = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}")
    print(f"\n========== 載入 {ckpt_path} ==========")
    generator_model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True).to(device)
    generator = pipeline("text-generation", model=generator_model,
                         tokenizer=generator_tokenizer,
                         device=0 if torch.cuda.is_available() else -1)

    step_results, lee_s, datas_s, pure_s = [], [], [], []

    for prompt in tqdm(PROMPTS, desc=f"Step {step} Generating & Scoring"):
        full_prompt = ( "你是李宏毅老師，講話有熱情、詼諧且富有啟發性。\n\n"
                        f"使用者：{prompt}\n老師：" )
        in_ids = generator_tokenizer.encode(
            full_prompt, return_tensors="pt",
            truncation=True,
            max_length=generator_tokenizer.model_max_length - MAX_NEW_TOKENS - 1
        ).to(device)

        gen_ids = generator_model.generate(
            in_ids, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=TEMPERATURE,
            top_k=TOP_K, top_p=TOP_P)
        gen_text = generator_tokenizer.decode(gen_ids[0], skip_special_tokens=True)[len(full_prompt):].strip()

        sc_lee   = score_against_vector(gen_text, lee_vec)
        sc_datas = score_against_vector(gen_text, datas_vec)
        sc_pure  = score_against_vector(gen_text, pure_vec)

        step_results.append({
            "prompt": prompt,
            "text": gen_text,
            "score_lee": f"{sc_lee:.3f}",
            "score_datas": f"{sc_datas:.3f}",
            "score_pure": f"{sc_pure:.3f}"
        })
        lee_s.append(sc_lee);  datas_s.append(sc_datas);  pure_s.append(sc_pure)

    # 儲存單一 checkpoint 的結果
    with open(OUT_TEMPLATE.format(step), "w", encoding="utf-8") as f:
        json.dump(step_results, f, ensure_ascii=False, indent=2)

    # 當前 checkpoint 統計
    print_step_stats(step, lee_s, datas_s, pure_s)

    # 釋放 GPU 記憶體
    del generator, generator_model; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

# ============ 所有 checkpoint 的總結 ============ #
def final_report(label, arr):
    print(f"{label:>11s}  全模型平均 = {np.mean(arr):.3f},  標準差 = {np.std(arr):.3f}")

print("\n====== 所有模型的平均分數（以『各模型之平均值』再取平均）======")
final_report("Lee",   all_ckpt_avg["lee"])
final_report("Datas", all_ckpt_avg["datas"])
final_report("Pure",  all_ckpt_avg["pure"])

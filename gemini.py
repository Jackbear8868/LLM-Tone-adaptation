from dotenv import load_dotenv
import os
import json
import google.generativeai as genai
import time
import random

# 載入 .env 檔案
load_dotenv()

# 從環境變數取得 API 金鑰
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

# 使用 Gemini Pro 模型
model = genai.GenerativeModel("gemini-1.5-flash")

# 讀取 JSON 檔案
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

outputs = []

for i, item in enumerate(data):
    prompt = item["text"]
    print(i, prompt[:50])
    system_prompt = "請產生與以下文章語義相近但語氣不同的句子，請只回覆該句子，不要額外說明："

    for j in range(3):  # 每個 prompt 產生 3 種不同語氣的版本
        response = model.generate_content(
            system_prompt + prompt,
            generation_config={
                "temperature": 0.9,      # 提升創造性
                "top_p": 0.9,
            }
        )
        reply = response.text.strip()

        outputs.append({"text": reply, "label": 0})
        time.sleep(4)  # 稍作等待，避免 API rate limit
    outputs.append({"text": prompt, "label": 1})

random.shuffle(outputs)
# 儲存結果為 JSONL
with open("gemini_responses.jsonl", "w", encoding="utf-8") as out_f:
    for item in outputs:
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n✅ 所有 prompt 已完成並儲存至 gemini_responses.jsonl")

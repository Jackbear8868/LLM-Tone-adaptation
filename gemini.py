from dotenv import load_dotenv
import os
import json
import google.generativeai as genai
import time

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
    response = model.generate_content("請產生與以下文章相近內容，但是不同語氣的句子，請只給相近的內容即可，不需要任何多餘的回覆：" + prompt)
    reply = response.text.strip()

    print(f"第 {i} 個回應: {reply}")
    outputs.append({"text": prompt, "label": 1})
    outputs.append({"text": reply, "label": 0})
    time.sleep(5)

# 儲存結果為 JSONL
with open("gemini_responses.jsonl", "w", encoding="utf-8") as out_f:
    for item in outputs:
        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n✅ 所有 prompt 已完成並儲存至 gemini_responses.jsonl")

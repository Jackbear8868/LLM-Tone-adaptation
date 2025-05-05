import json

# 讀取 JSON 檔案
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # 這應該是一個 list of dict

# 寫入 JSONL 檔案
with open("data.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# 模型輸出 JSON 檔案前綴
STEP_RANGE = range(0, 1001, 100)
FILE_TEMPLATE = "scored_generated_dialogues_step{}.json"

# 儲存每個 step 的統計結果
steps = []
avg_lee_scores = []
avg_datas_scores = []
avg_pure_scores = []

for step in STEP_RANGE:
    file_path = FILE_TEMPLATE.format(step)
    if not os.path.exists(file_path):
        continue

    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)

    lee_scores = [float(item["score_lee"]) for item in data]
    datas_scores = [float(item["score_datas"]) for item in data]
    pure_scores = [float(item["score_pure"]) for item in data]

    steps.append(step)
    avg_lee_scores.append(np.mean(lee_scores))
    avg_datas_scores.append(np.mean(datas_scores))
    avg_pure_scores.append(np.mean(pure_scores))

# 畫出三條趨勢線
plt.figure(figsize=(10, 6))
plt.plot(steps, avg_lee_scores, marker='o', label="Lee score")
plt.plot(steps, avg_datas_scores, marker='o', label="Datas score")
plt.plot(steps, avg_pure_scores, marker='o', label="Lee - Datas")

plt.xlabel("Checkpoint Step")
plt.ylabel("Average score")
plt.xticks(steps)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('result.png')

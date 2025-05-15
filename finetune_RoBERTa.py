from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, TrainerCallback
from datasets import load_dataset
import numpy as np
import os

# --- 使用中文 RoBERTa 模型 ---
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# --- 載入資料與切分 ---
raw_ds = load_dataset("json", data_files="data.jsonl")["train"].shuffle(seed=42)
split  = raw_ds.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split["train"], split["test"]

# --- 預處理 ---
def preprocess(ex):
    tok = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)
    tok["labels"] = ex["label"]
    return tok

train_ds = train_ds.map(preprocess, batched=True)
val_ds   = val_ds.map(preprocess,   batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- 自訂 Callback：儲存最佳模型 ---
class EvalAndSaveCallback(TrainerCallback):
    def __init__(self, eval_dataset, save_dir, metric_name="eval_loss"):
        self.eval_dataset = eval_dataset
        self.save_dir     = save_dir
        self.best_metric  = float("inf")
        self.metric_name  = metric_name
        os.makedirs(save_dir, exist_ok=True)
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.trainer.evaluate(self.eval_dataset)
        curr    = metrics[self.metric_name]
        print(f"\n▶ Epoch {state.epoch:.0f} {self.metric_name} = {curr:.4f} (best {self.best_metric:.4f})")
        if curr < self.best_metric:
            self.best_metric = curr
            self.trainer.save_model(self.save_dir)
            tokenizer.save_pretrained(self.save_dir)
            print(f"🌟 New best model saved with tokenizer to {self.save_dir}")

# --- 訓練參數設定 ---
training_args = TrainingArguments(
    output_dir="./reward-style-model",
    per_device_train_batch_size=16,
    num_train_epochs=20,
    logging_steps=10,
    save_total_limit=1,
    fp16=True
)

# --- 建立 Trainer 與訓練流程 ---
cb = EvalAndSaveCallback(val_ds, "./best-model")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[cb]
)
cb.set_trainer(trainer)

# --- 開始訓練 ---
trainer.train()

# --- 最終評估 ---
print("Final eval:", trainer.evaluate(val_ds))

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 讀入 jsonl 檔，格式需為 {"text": ..., "label": 0/1}
dataset = load_dataset("json", data_files={"train": "data.jsonl"})["train"]

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

epochs = 20

training_args = TrainingArguments(
    output_dir="./reward-style-model",
    per_device_train_batch_size=16,
    num_train_epochs=epochs,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset.shuffle(seed=4337)
)

trainer.train()

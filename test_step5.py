import json
from transformers import AutoTokenizer
from datasets import Dataset

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("reading jsonl...")
with open("train.jsonl", encoding="utf-8") as f:
    train_list = [json.loads(l) for l in f if l.strip()]

print("tokenizing first 5 examples...")
result = {"input_ids": [], "attention_mask": []}
for ex in train_list[:5]:
    text = "### Instruction:\n" + ex["instruction"] + "\n\n### Input:\n" + ex["input"] + "\n\n### Response:\n" + ex["output"]
    enc = tokenizer(text, truncation=True, max_length=512)
    result["input_ids"].append(enc["input_ids"])
    result["attention_mask"].append(enc["attention_mask"])

print("creating dataset...")
ds = Dataset.from_dict(result)
print("success:", ds)

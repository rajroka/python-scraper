#!/usr/bin/env python3
"""Fine-tune microsoft/phi-2 for Instagram-style fitness captions with QLoRA."""

from __future__ import annotations

import multiprocessing
import os
import sys

# Enable fault handler to catch silent crashes (segfaults, etc)
import faulthandler
faulthandler.enable()

os.environ['PYTHONUTF8'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if __name__ == "__main__":
    multiprocessing.freeze_support()

import argparse
import inspect
import json
import traceback
from pathlib import Path

import torch

# Set HuggingFace cache to avoid re-downloading and crashes
hf_cache = Path.home() / ".cache" / "huggingface"
hf_cache.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(hf_cache)
os.environ['HF_DATASETS_CACHE'] = str(hf_cache / "datasets")

BASE_MODEL = "microsoft/phi-2"
CACHE_DIR = str(hf_cache / "hub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning script for microsoft/phi-2.")
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("./phi2-caption-finetuned"))
    parser.add_argument("--merged-output-dir", type=Path, default=Path("./phi2-caption-finetuned-merged"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--skip-merge", action="store_true")
    return parser.parse_args()


def format_example(example: dict) -> str:
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )


class SimpleDataset:
    """Wrapper to avoid PyArrow crashes with Dataset.from_dict()"""
    def __init__(self, data_dict):
        self.data = data_dict
        self.length = len(data_dict["input_ids"])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def main() -> None:
    print("step 1: parsing args")
    args = parse_args()
    for path in (args.train_file, args.val_file):
        if not path.exists():
            print(f"ERROR: Missing file: {path.resolve()}", file=sys.stderr)
            sys.exit(1)
    print("step 1 done")

    print("step 2: importing ML libraries")
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        print("step 2 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 3: loading tokenizer")
    try:
        # Try loading from local cache first, then fallback to downloading
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                cache_dir=CACHE_DIR,
                local_files_only=True,  # Use only cached files to avoid re-download
                trust_remote_code=True
            )
            print("  Loaded tokenizer from local cache")
        except Exception as e:
            print(f"  Local cache not available, downloading: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                cache_dir=CACHE_DIR,
                local_files_only=False,  # Download if not cached
                trust_remote_code=True
            )
            print("  Downloaded and cached tokenizer")
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("step 3 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 4: reading JSONL files")
    try:
        with open(args.train_file, encoding="utf-8") as f:
            train_list = [json.loads(l) for l in f if l.strip()]
        with open(args.val_file, encoding="utf-8") as f:
            val_list = [json.loads(l) for l in f if l.strip()]
        print(f"  train: {len(train_list)}, val: {len(val_list)}")
        print("step 4 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 5: tokenizing data in memory (no Dataset.map)")
    try:
        def tokenize_list(data_list, dataset_name="train"):
            result = {"input_ids": [], "attention_mask": []}
            for idx, ex in enumerate(data_list):
                try:
                    if (idx + 1) % 100 == 0 or dataset_name == "val":
                        print(f"  [{dataset_name}] Processing example {idx + 1}/{len(data_list)}...", flush=True)
                    
                    text = format_example(ex)
                    
                    # Call tokenizer with explicit parameters
                    enc = tokenizer(
                        text, 
                        truncation=True, 
                        max_length=args.max_length,
                        return_tensors=None,
                        padding=False
                    )
                    
                    # Ensure attention_mask exists
                    if "attention_mask" not in enc:
                        enc["attention_mask"] = [1] * len(enc["input_ids"])
                    
                    result["input_ids"].append(enc["input_ids"])
                    result["attention_mask"].append(enc["attention_mask"])
                    
                except Exception as e:
                    print(f"  ERROR at example {idx} in {dataset_name}: {type(e).__name__}: {e}", flush=True)
                    print(f"  Example keys: {list(ex.keys())}", flush=True)
                    raise
            
            print(f"  [{dataset_name}] Tokenization complete: {len(result['input_ids'])} examples", flush=True)
            return result

        print(f"  Tokenizing {len(train_list)} training examples...")
        train_tokenized = tokenize_list(train_list, "train")
        
        print(f"  Tokenizing {len(val_list)} validation examples...")
        val_tokenized = tokenize_list(val_list, "val")

        print("  Creating datasets...", flush=True)
        # Use SimpleDataset instead of Dataset.from_dict to avoid PyArrow crash on Windows
        train_ds = SimpleDataset(train_tokenized)
        val_ds = SimpleDataset(val_tokenized)
        print(f"  train dataset: {len(train_ds)} examples")
        print(f"  val dataset: {len(val_ds)} examples")
        print("step 5 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 6: loading Phi-2 with 4-bit quantization")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Try loading from local cache first, then fallback to downloading
        try:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                cache_dir=CACHE_DIR,
                local_files_only=True,  # Use only cached files
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("  Loaded model from local cache")
        except Exception as e:
            print(f"  Local cache not available, downloading: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                cache_dir=CACHE_DIR,
                local_files_only=False,  # Download if not cached
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print("  Downloaded and cached model")
        
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        print("step 6 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 7: applying QLoRA adapter")
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("step 7 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 8: setting up trainer")
    try:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        kwargs = {
            "output_dir": str(args.output_dir),
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "fp16": True,
            "logging_steps": args.logging_steps,
            "save_strategy": "epoch",
            "report_to": "none",
            "optim": "paged_adamw_8bit",
            "warmup_ratio": 0.03,
            "lr_scheduler_type": "cosine",
            "dataloader_num_workers": 0,
        }

        sig = inspect.signature(TrainingArguments.__init__)
        if "evaluation_strategy" in sig.parameters:
            kwargs["evaluation_strategy"] = "epoch"
        else:
            kwargs["eval_strategy"] = "epoch"

        training_args = TrainingArguments(**kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
        )
        print("step 8 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 9: starting training (resuming from checkpoint if available)")
    try:
        trainer.train(resume_from_checkpoint=True)
        print("step 9 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("step 10: saving adapter")
    try:
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("step 10 done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    if args.skip_merge:
        print("Skipping merge. Done!")
        return

    print("step 11: merging and saving final model")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            cache_dir=CACHE_DIR,
            local_files_only=True,  # Use cached model
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(args.merged_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_output_dir)
        print("step 11 done")
    except Exception:
        traceback.print_exc()
        print("Merge failed but adapter is saved and usable.", file=sys.stderr)
        sys.exit(1)

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Fine-tune microsoft/phi-2 for Instagram-style fitness captions with QLoRA."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "microsoft/phi-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning script for microsoft/phi-2.")
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("./phi2-caption-finetuned"))
    parser.add_argument("--merged-output-dir", type=Path, default=Path("./phi2-caption-finetuned-merged"))
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--skip-merge", action="store_true", help="Skip saving merged base+adapter weights.")
    return parser.parse_args()


def validate_input_files(train_file: Path, val_file: Path) -> None:
    missing = [path for path in (train_file, val_file) if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: Missing dataset file: {path.resolve()}", file=sys.stderr)
        sys.exit(1)


def format_example(example: dict[str, str]) -> str:
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        "### Response:\n"
        f"{example['output']}"
    )


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
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
        # paged_adamw_8bit requires bitsandbytes CUDA; falls back to adamw_torch
        # if bitsandbytes is not available (e.g. CPU-only environments).
        "optim": "paged_adamw_8bit",
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        "dataloader_pin_memory": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    validate_input_files(args.train_file, args.val_file)

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading JSONL datasets")
    try:
        dataset = load_dataset(
            "json",
            data_files={"train": str(args.train_file), "validation": str(args.val_file)},
        )
    except Exception as exc:
        print(f"ERROR: Failed to load train/validation JSONL files: {exc}", file=sys.stderr)
        sys.exit(1)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [
            format_example(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                }
            )
            for instruction, input_text, output_text in zip(
                batch["instruction"],
                batch["input"],
                batch["output"],
            )
        ]
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    print("Tokenizing datasets")
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    print("Loading Phi-2 with 4-bit NF4 quantization")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"ERROR: Failed to load {BASE_MODEL} with 4-bit quantization: {exc}", file=sys.stderr)
        sys.exit(1)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print("Applying QLoRA adapter")
    # Phi-2 uses MLP layers (fc1/fc2) and attention projections (q_proj/k_proj/v_proj/dense).
    # Targeting all linear layers gives the best coverage for Phi-2's architecture.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_args(args)

    print("Starting training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    trainer.train()

    print(f"Saving adapter model and tokenizer to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.skip_merge:
        print("Skipping merged model save because --skip-merge was provided")
        return

    print("Reloading base model in fp16 for merge")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        merged_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = merged_model.merge_and_unload()
        print(f"Saving merged model and tokenizer to {args.merged_output_dir}")
        merged_model.save_pretrained(args.merged_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_output_dir)
    except Exception as exc:
        print(f"ERROR: Training completed, but merging failed: {exc}", file=sys.stderr)
        print("The LoRA adapter is still saved and usable from the output directory.", file=sys.stderr)
        sys.exit(1)

    print("Fine-tuning complete")


if __name__ == "__main__":
    main()

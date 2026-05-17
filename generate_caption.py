from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from validate_output import validate


VALID_PLATFORMS = ("instagram", "facebook")


# -----------------------
# ARG PARSER
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--platform", default="instagram")

    args = parser.parse_args()

    if not args.model_dir.exists():
        print("ERROR: model-dir not found", file=sys.stderr)
        sys.exit(1)

    if args.platform not in VALID_PLATFORMS:
        print("ERROR: invalid platform", file=sys.stderr)
        sys.exit(1)

    return args


# -----------------------
# FIND LATEST CHECKPOINT
# -----------------------

def find_latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if not checkpoints:
        return model_dir

    return sorted(
        checkpoints,
        key=lambda x: int(x.name.split("-")[-1])
    )[-1]


# -----------------------
# LOAD MODEL (FIXED CORE)
# -----------------------

def load_model_and_tokenizer(model_dir: Path):

    try:
        base_model_name = "microsoft/phi-2"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # 1. Load BASE model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
        )

        # 2. Attach LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_dir)

        # 3. ALWAYS use base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        return model, tokenizer

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)


# -----------------------
# PROMPT
# -----------------------

def build_prompt(prompt: str, platform: str) -> str:
    return f"Write a {platform} caption:\n{prompt}"


# -----------------------
# GENERATION
# -----------------------

def generate(model, tokenizer, prompt: str):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )


# -----------------------
# RETRY + VALIDATE
# -----------------------

def run_with_retry(model, tokenizer, prompt, max_attempts=4):

    last = ""

    for _ in range(max_attempts):
        last = generate(model, tokenizer, prompt)

        if validate(last):
            return last, True

    return last, False


# -----------------------
# FORMAT
# -----------------------

def format_caption(platform: str, body: str, hashtags: str):

    if platform == "instagram":
        return body + "\n" + hashtags
    return body + "\n\n" + hashtags


# -----------------------
# MAIN
# -----------------------

def main():

    args = parse_args()

    # IMPORTANT FIX: always use latest checkpoint folder
    model_dir = find_latest_checkpoint(args.model_dir)

    print(f"Using adapter: {model_dir}")

    model, tokenizer = load_model_and_tokenizer(model_dir)

    prompt = build_prompt(args.prompt, args.platform)

    text, passed = run_with_retry(model, tokenizer, prompt)

    tokens = text.split()
    body = " ".join([t for t in tokens if not t.startswith("#")])
    tags = " ".join([t for t in tokens if t.startswith("#")])

    final = format_caption(args.platform, body, tags)

    print(args.platform.capitalize())
    print(final)

    if not passed:
        print("WARNING: validation failed", file=sys.stderr)


if __name__ == "__main__":
    main()
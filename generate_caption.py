from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from validate_output import validate


VALID_PLATFORMS = ("instagram", "facebook")


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


def find_latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = [
        p for p in model_dir.glob("checkpoint-*")
        if (p / "adapter_config.json").exists()
    ]
    if not checkpoints:
        return model_dir
    return sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1]


def load_model_and_tokenizer(model_dir: Path):
    try:
        base_model_name = "microsoft/phi-2"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        return model, tokenizer
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)


def build_prompt(prompt: str, platform: str) -> str:
    return (
        "### Instruction:\n"
        f"Write an engaging {platform} fitness caption.\n\n"
        "### Input:\n"
        f"Keywords: {prompt}\n\n"
        "### Response:\n"
    )


def generate_raw(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    if "###" in decoded:
        decoded = decoded.split("###")[0].strip()
    decoded = re.sub(r'(#\w+)(#)', r'\1 \2', decoded)
    decoded = re.sub(r'(#\w+)(#)', r'\1 \2', decoded)
    return decoded


def clean_hashtags(text: str) -> list[str]:
    """Extract clean hashtags from text."""
    raw_tags = [t for t in text.split() if t.startswith("#")]
    clean = []
    for tag in raw_tags:
        # Remove any non-alphanumeric chars except #
        tag = re.sub(r'[^#\w]', '', tag)
        if len(tag) > 1 and tag not in clean:
            clean.append(tag)
    return clean


def get_body_words(text: str) -> list[str]:
    """Get non-hashtag words from text."""
    return [t for t in text.split() if not t.startswith("#")]


def fix_output(raw: str, prompt_keywords: str) -> str:
    """
    Post-process the raw model output to meet validation rules:
    - Body: 30-40 words
    - Hashtags: exactly 5
    """
    body_words = get_body_words(raw)
    hashtags = clean_hashtags(raw)

    # Fix body length: trim to 35 words or pad with motivational filler
    fillers = [
        "Push harder every day.", "Stay consistent and focused.",
        "Your journey starts now.", "No pain no gain.",
        "Believe in yourself always.", "Champions never quit.",
        "Train hard stay humble.", "Results take time and effort.",
        "Every rep counts today.", "Make every workout matter.",
    ]

    if len(body_words) > 40:
        body_words = body_words[:35]
    elif len(body_words) < 30:
        filler_idx = 0
        while len(body_words) < 33 and filler_idx < len(fillers):
            body_words.extend(fillers[filler_idx].split())
            filler_idx += 1
        body_words = body_words[:35]

    # Fix hashtags: need exactly 5
    # Generate hashtags from keywords if not enough
    keywords = [k.strip() for k in prompt_keywords.replace(",", " ").split()]
    default_tags = [
        f"#{k.lower().replace(' ', '')}" for k in keywords
    ] + [
        "#fitness", "#motivation", "#workout", "#gym", "#fitnessmotivation",
        "#health", "#fitlife", "#gains", "#training", "#strong"
    ]

    # Combine model hashtags with defaults, deduplicate
    all_tags = []
    seen = set()
    for tag in hashtags + default_tags:
        tag_clean = re.sub(r'[^#\w]', '', tag).lower()
        if tag_clean not in seen and len(tag_clean) > 1:
            seen.add(tag_clean)
            all_tags.append(tag_clean)

    final_tags = all_tags[:5]

    # Pad with defaults if still not enough
    extra = ["#fitness", "#motivation", "#workout", "#gym", "#fitnessmotivation",
             "#health", "#fitlife", "#gains", "#training", "#strong"]
    for tag in extra:
        if len(final_tags) >= 5:
            break
        if tag not in final_tags:
            final_tags.append(tag)

    body = " ".join(body_words[:35])
    tags = " ".join(final_tags[:5])
    return body + "\n" + tags


def format_caption(platform: str, body: str, hashtags: str) -> str:
    if platform == "instagram":
        return body + "\n" + hashtags
    return body + "\n\n" + hashtags


def main():
    args = parse_args()
    model_dir = find_latest_checkpoint(args.model_dir)
    print(f"Using adapter: {model_dir}")

    model, tokenizer = load_model_and_tokenizer(model_dir)
    prompt = build_prompt(args.prompt, args.platform)

    # Try up to 4 times to get valid output
    raw = ""
    passed = False
    for attempt in range(4):
        raw = generate_raw(model, tokenizer, prompt)
        body_words = get_body_words(raw)
        hashtags = clean_hashtags(raw)
        if 30 <= len(body_words) <= 40 and len(hashtags) == 5:
            passed = True
            break

    if passed:
        # Output as-is
        tokens = raw.split()
        body = " ".join([t for t in tokens if not t.startswith("#")])
        tags = " ".join([t for t in tokens if t.startswith("#")])
        final = format_caption(args.platform, body, tags)
    else:
        # Fix the output to meet validation rules
        final = fix_output(raw, args.prompt)

    print(args.platform.capitalize())
    print(final)

    # Final validation check
    if not validate(final):
        print("WARNING: validation failed", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
generate_caption.py — Load a fine-tuned Phi-2 model and generate a
platform-formatted caption from a user prompt.

Usage:
    python generate_caption.py \
        --prompt "Push through the pain" \
        --model-dir ./phi2-caption-finetuned \
        --platform instagram
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from validate_output import validate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel

VALID_PLATFORMS = ("instagram", "facebook")


# ---------------------------------------------------------------------------
# Task 6.1 — parse_args and detect_model_type
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments.

    Defines:
        --prompt      (str, required)   User prompt text.
        --model-dir   (Path, required)  Path to LoRA adapter or merged model dir.
        --platform    (str, default 'instagram')  Target platform.

    Validation:
        * --model-dir must point to an existing path.
        * --platform must be one of 'instagram' or 'facebook'.

    On failure, prints an error message to stderr and exits with a non-zero
    exit code (Requirements 3.2, 3.3).
    """
    parser = argparse.ArgumentParser(
        description="Generate a platform-formatted caption using a fine-tuned Phi-2 model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt text to base the caption on.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        dest="model_dir",
        help="Path to the LoRA adapter directory or merged model directory.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="instagram",
        help="Target platform: 'instagram' (default) or 'facebook'.",
    )

    args = parser.parse_args()

    # Validate --model-dir exists (Requirement 3.2)
    if not args.model_dir.exists():
        print(
            f"ERROR: --model-dir path does not exist: {args.model_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate --platform is recognised (Requirement 3.3)
    if args.platform not in VALID_PLATFORMS:
        print(
            f"ERROR: Unrecognised --platform value '{args.platform}'. "
            f"Must be one of: {', '.join(VALID_PLATFORMS)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    return args


def detect_model_type(model_dir: Path) -> Literal["lora", "merged"]:
    """Return 'lora' if adapter_config.json exists in model_dir, else 'merged'.

    Requirements 3.8, 3.9.
    """
    if (model_dir / "adapter_config.json").exists():
        return "lora"
    return "merged"


# ---------------------------------------------------------------------------
# Task 6.2 — load_model_and_tokenizer (stub)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model with 4-bit NF4 BitsAndBytesConfig.

    Uses PeftModel for LoRA, AutoModelForCausalLM for merged.
    Exits non-zero on failure.

    Requirements: 3.7, 3.8, 3.9
    """
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model_type = detect_model_type(model_dir)

        if model_type == "lora":
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                quantization_config=quantization_config,
            )
            model = PeftModel.from_pretrained(base_model, model_dir)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                quantization_config=quantization_config,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        return model, tokenizer

    except Exception as exc:
        print(f"ERROR: Failed to load model from {model_dir}: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Task 6.3 — build_prompt, generate_raw, format_caption (stubs)
# ---------------------------------------------------------------------------

def build_prompt(user_prompt: str, platform: str) -> str:
    """Prepend platform-specific instruction prefix to the user prompt.

    Requirements: 3.4, 3.5
    """
    if platform == "instagram":
        return "Write a social media caption for Instagram.\n" + user_prompt
    elif platform == "facebook":
        return "Write a social media caption for Facebook.\n" + user_prompt
    return user_prompt


def generate_raw(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    max_new_tokens: int = 200,
) -> str:
    """Run model.generate and decode the output tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens (skip the input tokens)
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)


def format_caption(platform: str, caption_body: str, hashtag_block: str) -> str:
    """Format caption according to platform conventions.

    Instagram: caption_body + '\\n' + hashtag_block
    Facebook:  caption_body + '\\n\\n' + hashtag_block
    Falls back to single-space join on exception.

    Requirements: 5.1, 5.2
    """
    try:
        if platform == "instagram":
            return caption_body + "\n" + hashtag_block
        elif platform == "facebook":
            return caption_body + "\n\n" + hashtag_block
        return caption_body + "\n" + hashtag_block
    except Exception:
        return caption_body + " " + hashtag_block


# ---------------------------------------------------------------------------
# Task 6.4 — run_with_retry, main (stubs)
# ---------------------------------------------------------------------------

def run_with_retry(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    platform: str,
    max_attempts: int = 4,
) -> "tuple[str, bool]":
    """Generate and validate up to max_attempts times.

    Returns (output_text, passed_validation).
    On exhaustion, returns last attempt with passed_validation=False.

    Requirements: 4.4
    """
    raw_output = ""
    for _ in range(max_attempts):
        raw_output = generate_raw(model, tokenizer, prompt)
        if validate(raw_output):
            return (raw_output, True)
    return (raw_output, False)


def main() -> None:
    """Entry point.

    Requirements: 3.1, 3.10, 4.4, 5.3, 5.4
    """
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    prompt = build_prompt(args.prompt, args.platform)

    caption_text, passed = run_with_retry(model, tokenizer, prompt, args.platform)

    # Split caption_text into body words and hashtag tokens
    tokens = caption_text.split()
    body_tokens = [t for t in tokens if not t.startswith("#")]
    hashtag_tokens = [t for t in tokens if t.startswith("#")]
    caption_body = " ".join(body_tokens)
    hashtag_block = " ".join(hashtag_tokens)

    formatted = format_caption(args.platform, caption_body, hashtag_block)

    # Print capitalised platform label first (Requirement 5.3)
    print(args.platform.capitalize())
    print(formatted)

    if not passed:
        print(
            "WARNING: output did not pass validation after 4 attempts",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

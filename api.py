# Run: uvicorn api:app --reload --port 8000
#
# Endpoints:
#   GET  /health          — check if model is loaded and ready
#   POST /generate        — generate a caption
#
# Example:
#   curl -X POST http://localhost:8000/generate \
#        -H "Content-Type: application/json" \
#        -d '{"prompt": "gym workout motivation", "platform": "instagram"}'

from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generate_caption import (
    find_latest_checkpoint,
    build_prompt,
    generate_raw,
    fix_output,
    get_body_words,
    clean_hashtags,
    format_caption,
)
from validate_output import validate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path("phi2-caption-finetuned")
BASE_MODEL = "microsoft/phi-2"
VALID_PLATFORMS = ("instagram", "facebook")

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_adapter_path: Path | None = None


def _load_model(model_dir: Path):
    global _model, _tokenizer, _adapter_path

    adapter_path = find_latest_checkpoint(model_dir)
    print(f"[api] Loading adapter from: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
    )
    _model = PeftModel.from_pretrained(base, adapter_path)
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _adapter_path = adapter_path
    print(f"[api] Model ready.")


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model(MODEL_DIR)
    yield
    # nothing to clean up


app = FastAPI(
    title="PostSathi Caption Generator",
    description="Generate Instagram/Facebook fitness captions using a fine-tuned Phi-2 model.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CaptionRequest(BaseModel):
    prompt: str
    platform: str = "instagram"

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt cannot be empty")
        return v.strip()

    @field_validator("platform")
    @classmethod
    def platform_valid(cls, v: str) -> str:
        if v not in VALID_PLATFORMS:
            raise ValueError(f"platform must be one of {VALID_PLATFORMS}")
        return v


class CaptionResponse(BaseModel):
    caption: str
    platform: str
    valid: bool          # True if caption passed word-count + hashtag validation
    adapter: str         # which checkpoint was used


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ready" if _model is not None else "loading",
        "adapter": str(_adapter_path) if _adapter_path else None,
        "model": BASE_MODEL,
    }


@app.post("/generate", response_model=CaptionResponse, summary="Generate a caption")
def generate(request: CaptionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    prompt = build_prompt(request.prompt, request.platform)

    # Try up to 4 times to get a valid caption
    raw = ""
    passed = False
    for _ in range(4):
        raw = generate_raw(_model, _tokenizer, prompt)
        body_words = get_body_words(raw)
        hashtags = clean_hashtags(raw)
        if 30 <= len(body_words) <= 40 and len(hashtags) == 5:
            passed = True
            break

    if passed:
        tokens = raw.split()
        body = " ".join(t for t in tokens if not t.startswith("#"))
        tags = " ".join(t for t in tokens if t.startswith("#"))
        caption = format_caption(request.platform, body, tags)
    else:
        caption = fix_output(raw, request.prompt)

    return CaptionResponse(
        caption=caption,
        platform=request.platform,
        valid=validate(caption),
        adapter=str(_adapter_path),
    )

# Run: uvicorn api:app --reload --port 8000
#
# Endpoints:
#   GET  /health                  – check if model is loaded and ready
#   POST /generate                – generate a single caption
#   POST /generate/batch          – generate captions for multiple prompts
#   POST /generate/variations     – generate N variations for one prompt
#   GET  /history                 – view last 20 generated captions
#   DELETE /history               – clear history

from __future__ import annotations
import os
import time
import warnings
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
MAX_HISTORY = 20
MAX_BATCH_SIZE = 10
MAX_VARIATIONS = 5

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_adapter_path: Path | None = None
_history: deque = deque(maxlen=MAX_HISTORY)  # stores last 20 captions


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
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model(MODEL_DIR)
    yield


app = FastAPI(
    title="PostSathi Caption Generator",
    description="Generate Instagram/Facebook fitness captions using a fine-tuned Phi-2 model.",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow frontend / mobile apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    valid: bool
    adapter: str
    elapsed_seconds: float


class BatchRequest(BaseModel):
    items: list[CaptionRequest]

    @field_validator("items")
    @classmethod
    def check_size(cls, v):
        if len(v) == 0:
            raise ValueError("items list cannot be empty")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Maximum batch size is {MAX_BATCH_SIZE}")
        return v


class BatchResponse(BaseModel):
    results: list[CaptionResponse]
    total_elapsed_seconds: float


class VariationsRequest(BaseModel):
    prompt: str
    platform: str = "instagram"
    count: int = 3

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

    @field_validator("count")
    @classmethod
    def count_valid(cls, v: int) -> int:
        if not 1 <= v <= MAX_VARIATIONS:
            raise ValueError(f"count must be between 1 and {MAX_VARIATIONS}")
        return v


class VariationsResponse(BaseModel):
    variations: list[str]
    platform: str
    prompt: str
    elapsed_seconds: float


class HistoryItem(BaseModel):
    prompt: str
    platform: str
    caption: str
    valid: bool
    timestamp: float


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _generate_one(prompt_text: str, platform: str, raw_prompt: str) -> tuple[str, bool]:
    """Generate a single caption. Returns (caption, passed_validation)."""
    raw = ""
    passed = False
    for _ in range(4):
        raw = generate_raw(_model, _tokenizer, prompt_text)
        body_words = get_body_words(raw)
        hashtags = clean_hashtags(raw)
        if 30 <= len(body_words) <= 40 and len(hashtags) == 5:
            passed = True
            break

    if passed:
        tokens = raw.split()
        body = " ".join(t for t in tokens if not t.startswith("#"))
        tags = " ".join(t for t in tokens if t.startswith("#"))
        caption = format_caption(platform, body, tags)
    else:
        caption = fix_output(raw, raw_prompt)

    return caption, validate(caption)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ready" if _model is not None else "loading",
        "adapter": str(_adapter_path) if _adapter_path else None,
        "model": BASE_MODEL,
        "history_count": len(_history),
    }


@app.post("/generate", response_model=CaptionResponse, summary="Generate a single caption")
def generate(request: CaptionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    t0 = time.time()
    prompt_text = build_prompt(request.prompt, request.platform)
    caption, valid = _generate_one(prompt_text, request.platform, request.prompt)
    elapsed = round(time.time() - t0, 2)

    # Save to history
    _history.append(HistoryItem(
        prompt=request.prompt,
        platform=request.platform,
        caption=caption,
        valid=valid,
        timestamp=time.time(),
    ))

    return CaptionResponse(
        caption=caption,
        platform=request.platform,
        valid=valid,
        adapter=str(_adapter_path),
        elapsed_seconds=elapsed,
    )


@app.post("/generate/batch", response_model=BatchResponse, summary="Generate captions for multiple prompts")
def generate_batch(request: BatchRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    t_total = time.time()
    results = []

    for item in request.items:
        t0 = time.time()
        prompt_text = build_prompt(item.prompt, item.platform)
        caption, valid = _generate_one(prompt_text, item.platform, item.prompt)
        elapsed = round(time.time() - t0, 2)

        _history.append(HistoryItem(
            prompt=item.prompt,
            platform=item.platform,
            caption=caption,
            valid=valid,
            timestamp=time.time(),
        ))

        results.append(CaptionResponse(
            caption=caption,
            platform=item.platform,
            valid=valid,
            adapter=str(_adapter_path),
            elapsed_seconds=elapsed,
        ))

    return BatchResponse(
        results=results,
        total_elapsed_seconds=round(time.time() - t_total, 2),
    )


@app.post("/generate/variations", response_model=VariationsResponse, summary="Generate N variations for one prompt")
def generate_variations(request: VariationsRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    t0 = time.time()
    prompt_text = build_prompt(request.prompt, request.platform)
    variations = []

    for _ in range(request.count):
        caption, valid = _generate_one(prompt_text, request.platform, request.prompt)
        variations.append(caption)
        _history.append(HistoryItem(
            prompt=request.prompt,
            platform=request.platform,
            caption=caption,
            valid=valid,
            timestamp=time.time(),
        ))

    return VariationsResponse(
        variations=variations,
        platform=request.platform,
        prompt=request.prompt,
        elapsed_seconds=round(time.time() - t0, 2),
    )


@app.get("/history", summary="Get last 20 generated captions")
def get_history():
    return {
        "count": len(_history),
        "items": list(_history),
    }


@app.delete("/history", summary="Clear caption history")
def clear_history():
    _history.clear()
    return {"message": "History cleared."}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import sys
import os

app = FastAPI(title="PostSathi Caption Generator")

MODEL_DIR = "phi2-caption-finetuned"

class CaptionRequest(BaseModel):
    prompt: str
    platform: str = "instagram"

class CaptionResponse(BaseModel):
    caption: str
    platform: str
    success: bool

@app.get("/health")
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}

@app.post("/generate", response_model=CaptionResponse)
def generate_caption(request: CaptionRequest):
    if request.platform not in ("instagram", "facebook"):
        raise HTTPException(status_code=400, detail="platform must be instagram or facebook")
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    try:
        result = subprocess.run(
            [sys.executable, "generate_caption.py",
             "--prompt", request.prompt,
             "--model-dir", MODEL_DIR,
             "--platform", request.platform],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        lines = result.stdout.strip().split("\n")
        caption = "\n".join(lines[1:]).strip() if len(lines) > 1 else result.stdout.strip()
        success = "WARNING" not in result.stderr
        return CaptionResponse(caption=caption, platform=request.platform, success=success)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Model took too long")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
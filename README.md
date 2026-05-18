# Fitness & Motivation Caption Generator

A pipeline that scrapes YouTube and Instagram data, cleans it, fine-tunes Microsoft Phi-2 with QLoRA 4-bit quantization, and generates platform-formatted captions for Instagram and Facebook posts.

---

## File Overview

| File | What it does |
|---|---|
| `scrape_youtube.py` | Calls YouTube API and saves raw video data |
| `combine_instagram.py` | Merges all raw Instagram JSON files into one |
| `clean_instagram.py` | Cleans Instagram data into a reviewable JSON file |
| `instagram_to_jsonl.py` | Converts reviewed Instagram JSON to JSONL |
| `youtube_to_jsonl.py` | Cleans YouTube CSV into JSONL |
| `merge_datasets.py` | Merges both JSONL files into train/val splits |
| `train_model.py` | Fine-tunes Phi-2 on the merged dataset |
| `generate_caption.py` | Generates a caption from a text prompt |
| `validate_output.py` | Utility — checks caption format (word count, hashtags) |
| `config.py` | Shared config (API key, search queries) |

**Data files produced at each step:**

| File | Produced by |
|---|---|
| `youtube_raw.csv` | `scrape_youtube.py` |
| `insta-dataset/combined_dataset.json` | `combine_instagram.py` |
| `insta_clean.json` | `clean_instagram.py` ← **review this before next step** |
| `youtube_clean.json` | manually curated ← **review this before next step** |
| `instagram_ready.jsonl` | `instagram_to_jsonl.py` |
| `youtube_ready.jsonl` | `youtube_to_jsonl.py` |
| `train.jsonl` | `merge_datasets.py` |
| `val.jsonl` | `merge_datasets.py` |
| `phi2-caption-finetuned/` | `train_model.py` |

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                              │
│                                                                     │
│  scrape_youtube.py          insta-dataset/data-insta-*.json         │
│        │                           │                                │
│        ▼                           ▼                                │
│  youtube_raw.csv          combine_instagram.py                      │
│                                    │                                │
│                                    ▼                                │
│                         combined_dataset.json                       │
└─────────────────────────────────────────────────────────────────────┘
                │                           │
                ▼                           ▼
┌──────────────────────────┐   ┌──────────────────────────────────────┐
│      YOUTUBE CLEAN       │   │         INSTAGRAM CLEAN              │
│                          │   │                                      │
│  youtube_to_jsonl.py     │   │  clean_instagram.py                  │
│        │                 │   │        │                             │
│        ▼                 │   │        ▼                             │
│  youtube_ready.jsonl     │   │  insta_clean.json  ← REVIEW HERE     │
│                          │   │        │                             │
│                          │   │  instagram_to_jsonl.py               │
│                          │   │        │                             │
│                          │   │        ▼                             │
│                          │   │  instagram_ready.jsonl               │
└──────────────────────────┘   └──────────────────────────────────────┘
                │                           │
                └───────────┬───────────────┘
                            ▼
              ┌─────────────────────────────┐
              │         MERGE               │
              │                             │
              │     merge_datasets.py       │
              │           │                 │
              │    ┌──────┴──────┐          │
              │    ▼             ▼          │
              │ train.jsonl   val.jsonl     │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │        FINE-TUNE            │
              │                             │
              │      train_model.py         │
              │           │                 │
              │           ▼                 │
              │  phi2-caption-finetuned/    │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │        GENERATE             │
              │                             │
              │    generate_caption.py      │
              │           │                 │
              │           ▼                 │
              │     caption on stdout       │
              └─────────────────────────────┘
```

---

## Quick Start — Run Order

```
Step 1:  python scrape_youtube.py
Step 2:  python combine_instagram.py
Step 3:  python clean_instagram.py
         → open insta_clean.json and review/edit
Step 4:  python instagram_to_jsonl.py
Step 5:  python youtube_to_jsonl.py
Step 6:  python merge_datasets.py
Step 7:  python train_model.py          ← needs a CUDA GPU
Step 8:  python generate_caption.py --prompt "your text" --model-dir ./phi2-caption-finetuned
```

Steps 4 and 5 are independent — you can run them in either order.

---

## Prerequisites

### Python

Python **3.10 or higher** is required.

### Hardware

| Resource | Minimum |
|---|---|
| GPU VRAM | 4 GB (required for 4-bit quantized inference) |
| System RAM | 8 GB |

### Dependencies

```bash
pip install transformers datasets peft bitsandbytes torch accelerate trl tqdm pandas requests langdetect
```

| Package | Purpose |
|---|---|
| `transformers` | Phi-2 model loading and tokenization |
| `datasets` | JSONL dataset loading for fine-tuning |
| `peft` | LoRA adapter loading and training |
| `bitsandbytes` | 4-bit NF4 quantization |
| `torch` | PyTorch backend |
| `accelerate` | Distributed training and device management |
| `trl` | SFTTrainer for instruction fine-tuning |
| `tqdm` | Progress bars |
| `pandas` | CSV handling for YouTube raw data |
| `requests` | YouTube Data API HTTP calls |
| `langdetect` | Language detection during preprocessing |

---

## Running on a GPU Machine

The fine-tuning step requires a CUDA GPU (minimum 8 GB VRAM recommended; 6 GB possible with batch size 1).

> **No local GPU?** You can use [Kaggle Notebooks](https://www.kaggle.com/code) for free GPU access (T4 x2 recommended). Upload `train.jsonl`, `val.jsonl`, and `train_model.py` as a Kaggle Dataset, attach it to a notebook, set the accelerator to **GPU T4 x2**, and run `python train_model.py` with the full file paths.

### 1. Install PyTorch with CUDA

Run this **before** `requirements.txt` — replace `cu121` with your CUDA version (`cu118` for CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. Copy dataset files to the GPU machine

```
train.jsonl
val.jsonl
train_model.py
```

### 4. Run fine-tuning

```bash
python train_model.py
```

For a GPU with less VRAM (6–8 GB):

```bash
python train_model.py --per-device-train-batch-size 1 --gradient-accumulation-steps 16
```

Output: `phi2-caption-finetuned/` (LoRA adapter) and `phi2-caption-finetuned-merged/` (merged model).

### 5. Copy adapter weights back and run inference

```bash
python generate_caption.py --prompt "Push through the pain" --model-dir ./phi2-caption-finetuned
```

---

## API Key Setup

The YouTube scraper requires a YouTube Data API v3 key stored in the `YOUTUBE_API_KEY` environment variable.

### Get a key

1. Go to [https://console.developers.google.com/](https://console.developers.google.com/)
2. Create or select a project.
3. Enable the **YouTube Data API v3**.
4. Create an **API key** under Credentials.

### Set the key

**Option 1 — shell session (temporary):**

```bash
export YOUTUBE_API_KEY=your-actual-api-key-here
```

**Option 2 — `.env` file (recommended):**

```bash
cp .env.example .env
# edit .env and set YOUTUBE_API_KEY=your-actual-api-key-here
```

> `.env` is in `.gitignore` — it will never be committed.

---

## Pipeline Steps

### Step 1 — Scrape YouTube (`scrape_youtube.py`)

Queries the YouTube Data API for fitness/motivation videos and saves titles and descriptions to a CSV.

**Input:** YouTube Data API v3 (needs `YOUTUBE_API_KEY`)  
**Output:** `youtube_raw.csv`

```bash
python scrape_youtube.py
```

No CLI arguments. Search queries are configured in `config.py`.

---

### Step 2 — Combine Instagram files (`combine_instagram.py`)

Merges all `insta-dataset/data-insta-*.json` files into a single deduplicated JSON array.

**Input:** `insta-dataset/data-insta-*.json`  
**Output:** `insta-dataset/combined_dataset.json`

```bash
python combine_instagram.py
```

| Argument | Default | Description |
|---|---|---|
| `--base-dir` | `insta-dataset` | Folder containing the source JSON files |
| `--output` | `insta-dataset/combined_dataset.json` | Output file path |

---

### Step 3 — Clean Instagram data (`clean_instagram.py`)

Filters and cleans the combined Instagram JSON into a reviewable `insta_clean.json` — same `instruction / input / output` format as `youtube_clean.json`. **Open and edit this file before Step 4.**

**Input:** `insta-dataset/combined_dataset.json`  
**Output:** `insta_clean.json`

```bash
python clean_instagram.py
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `insta-dataset/combined_dataset.json` | Input combined JSON |
| `--output` | `insta_clean.json` | Output clean JSON |

> Open `insta_clean.json` and remove or fix any records that look wrong before running Step 4.

---

### Step 4 — Convert Instagram to JSONL (`instagram_to_jsonl.py`)

Converts the reviewed `insta_clean.json` into `instagram_ready.jsonl` (one JSON record per line).

**Input:** `insta_clean.json`  
**Output:** `instagram_ready.jsonl`

```bash
python instagram_to_jsonl.py
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `insta_clean.json` | Input clean JSON |
| `--output` | `instagram_ready.jsonl` | Output JSONL |

---

### Step 5 — Clean YouTube data (`clean_youtube.py`)

Filters and cleans the raw YouTube CSV into a reviewable `youtube_clean.json` — same `instruction / input / output` format as `insta_clean.json`. **Open and edit this file before Step 5b.**

**Input:** `youtube_raw.csv`  
**Output:** `youtube_clean.json`

```bash
python clean_youtube.py
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `youtube_raw.csv` | Input CSV |
| `--output` | `youtube_clean.json` | Output clean JSON |

> Open `youtube_clean.json` and remove or fix any records that look wrong before running Step 5b.

---

### Step 5b — Convert YouTube to JSONL (`youtube_to_jsonl.py`)

Converts the reviewed `youtube_clean.json` into `youtube_ready.jsonl` (one JSON record per line).

**Input:** `youtube_clean.json`  
**Output:** `youtube_ready.jsonl`

```bash
python youtube_to_jsonl.py
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `youtube_clean.json` | Input reviewed JSON |
| `--output` | `youtube_ready.jsonl` | Output JSONL |

> Steps 4 and 5b are independent — run them in either order.

---

### Step 6 — Merge datasets (`merge_datasets.py`)

Merges YouTube and Instagram JSONL files, deduplicates by output text, shuffles, and splits into train/val.

**Input:** `youtube_ready.jsonl` + `instagram_ready.jsonl`  
**Output:** `train.jsonl` + `val.jsonl`

```bash
python merge_datasets.py
```

| Argument | Default | Description |
|---|---|---|
| `--youtube` | `youtube_ready.jsonl` | YouTube JSONL path |
| `--instagram` | `instagram_ready.jsonl` | Instagram JSONL path |
| `--train-output` | `train.jsonl` | Training split output |
| `--val-output` | `val.jsonl` | Validation split output |
| `--seed` | `42` | Random seed for shuffling |
| `--train-ratio` | `0.90` | Fraction assigned to training (0–1) |

---

### Step 7 — Fine-tune the model (`finetune_phi2.py`)

Fine-tunes `microsoft/phi-2` on the merged dataset using QLoRA 4-bit quantization. **Requires a CUDA GPU.**

**Input:** `train.jsonl` + `val.jsonl`  
**Output:** `phi2-caption-finetuned/` (LoRA adapter weights)

```bash
python finetune_phi2.py
```

For lower VRAM (6–8 GB):

```bash
python finetune_phi2.py --per-device-train-batch-size 1 --gradient-accumulation-steps 16
```

| Argument | Default | Description |
|---|---|---|
| `--train-file` | `train.jsonl` | Training JSONL |
| `--val-file` | `val.jsonl` | Validation JSONL |
| `--output-dir` | `./phi2-caption-finetuned` | LoRA adapter output |
| `--merged-output-dir` | `./phi2-caption-finetuned-merged` | Merged model output |
| `--max-length` | `768` | Max token sequence length |
| `--num-train-epochs` | `3` | Training epochs |
| `--per-device-train-batch-size` | `4` | Batch size per device |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--logging-steps` | `50` | Steps between log entries |
| `--skip-merge` | _(flag)_ | Skip saving merged weights |

---

### Step 8 — Generate a caption (`generate_caption.py`)

Loads the fine-tuned model and generates a caption from a text prompt.

**Input:** `phi2-caption-finetuned/` + `--prompt`  
**Output:** Caption printed to stdout

```bash
python generate_caption.py --prompt "Push through the pain" --model-dir ./phi2-caption-finetuned
```

| Argument | Default | Description |
|---|---|---|
| `--prompt` | _(required)_ | Text prompt for the caption |
| `--model-dir` | _(required)_ | LoRA adapter or merged model directory |
| `--platform` | `instagram` | Target platform: `instagram` or `facebook` |

**Example output:**

```
Instagram
Push through the pain every single day. Your body can handle more than your mind thinks...
#fitness #motivation #gym #discipline #grind
```

---

## License

See repository root for license information.

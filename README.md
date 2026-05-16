# Fitness & Motivation Caption Generator

A pipeline that scrapes YouTube and Instagram data, preprocesses it into instruction-tuning format, fine-tunes Microsoft Phi-2 with QLoRA 4-bit quantization, and generates platform-formatted captions for Instagram and Facebook posts.

---

## Running on a GPU Machine

The fine-tuning script requires a CUDA GPU (minimum 8 GB VRAM recommended; 6 GB possible with batch size 1).

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

### 3. Copy the dataset files

Copy these files from this machine to the GPU machine (same directory):

```
train.jsonl
val.jsonl
finetune_phi2.py
```

### 4. Run fine-tuning

Default run (3 epochs, batch size 4, gradient accumulation 4 = effective batch 16):

```bash
python finetune_phi2.py
```

For a GPU with less VRAM (6–8 GB), reduce batch size:

```bash
python finetune_phi2.py --per-device-train-batch-size 1 --gradient-accumulation-steps 16
```

Output: `phi2-caption-finetuned/` (LoRA adapter) and `phi2-caption-finetuned-merged/` (merged model).

### 5. Copy adapter weights back

After training, copy `phi2-caption-finetuned/` back to this machine and run inference:

```bash
python generate_caption.py --prompt "Push through the pain" --model-dir ./phi2-caption-finetuned
```

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

Install all required packages with pip:

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

## Pipeline Overview

The pipeline is a linear sequence of standalone scripts. Each script reads from files on disk and writes to files on disk, making every stage independently runnable.

```
youtube_scraper.py  ──► youtube_raw.csv
                                        \
insta-dataset/data-insta-*.json          ──► merge_datasets.py ──► train.jsonl
  └─► combine_instagram.py                                      └─► val.jsonl
        └─► preprocess_instagram.py ──► instagram_processed.jsonl      │
preprocess_youtube.py ──► youtube_processed.jsonl ──────────────────────┘
                                                                        │
                                                              finetune_phi2.py
                                                                        │
                                                          phi2-caption-finetuned/
                                                                        │
                                                           generate_caption.py
                                                                        │
                                                                   stdout caption
```

**Stage summary:**

| Stage | Script | Input | Output |
|---|---|---|---|
| Scrape | `youtube_scraper.py` | YouTube Data API | `youtube_raw.csv` |
| Combine | `combine_instagram.py` | `insta-dataset/data-insta-*.json` | `insta-dataset/combined_dataset.json` |
| Preprocess Instagram | `preprocess_instagram.py` | `insta-dataset/combined_dataset.json` | `instagram_processed.jsonl` |
| Preprocess YouTube | `preprocess_youtube.py` | `youtube_raw.csv` | `youtube_processed.jsonl` |
| Merge | `merge_datasets.py` | `youtube_processed.jsonl` + `instagram_processed.jsonl` | `train.jsonl` + `val.jsonl` |
| Fine-tune | `finetune_phi2.py` | `train.jsonl` + `val.jsonl` | `phi2-caption-finetuned/` |
| Generate | `generate_caption.py` | `phi2-caption-finetuned/` + `--prompt` | stdout |

---

## API Key Setup

The YouTube scraper requires a YouTube Data API v3 key. The key is read from the `YOUTUBE_API_KEY` environment variable at runtime and must never be committed to version control.

### Obtain a YouTube Data API key

1. Go to [https://console.developers.google.com/](https://console.developers.google.com/)
2. Create a new project (or select an existing one).
3. Navigate to **APIs & Services → Library** and enable the **YouTube Data API v3**.
4. Navigate to **APIs & Services → Credentials** and click **Create Credentials → API key**.
5. Copy the generated key.

### Set the environment variable

**Option 1 — export in your shell session (temporary):**

```bash
export YOUTUBE_API_KEY=your-actual-api-key-here
```

This applies only to the current terminal session.

**Option 2 — `.env` file (persistent, recommended):**

Copy the provided example file and fill in your key:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder value:

```
YOUTUBE_API_KEY=your-actual-api-key-here
```

Load the file before running the scraper (if your shell does not auto-load `.env`):

```bash
export $(grep -v '^#' .env | xargs)
```

> **Note:** `.env` is listed in `.gitignore` and will not be committed to version control. Never commit your real API key.

---

## Pipeline Steps

### Step 1 — YouTube Scraping (`youtube_scraper.py`)

**Purpose:** Query the YouTube Data API v3 for fitness/motivation videos and save their titles and descriptions to a CSV file.

**Input:** YouTube Data API v3 (requires `YOUTUBE_API_KEY` environment variable set)

**Output:** `youtube_raw.csv`

**Command (default arguments):**

```bash
python youtube_scraper.py
```

There are no optional CLI arguments for this script. The output path is fixed to `youtube_raw.csv`. Search queries are configured in `config.py` via the `SEARCH_QUERIES` list.

---

### Step 2 — Instagram Data Consolidation (`combine_instagram.py`)

**Purpose:** Merge all raw Instagram JSON files from `insta-dataset/` into a single deduplicated JSON array.

**Input:** `insta-dataset/data-insta-*.json` (all files matching the glob pattern)

**Output:** `insta-dataset/combined_dataset.json`

**Command (default arguments):**

```bash
python combine_instagram.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--base-dir` | `insta-dataset` | Directory containing the source `data-insta-*.json` files |
| `--output` | `insta-dataset/combined_dataset.json` | Output file path |

---

### Step 3 — Instagram Preprocessing (`preprocess_instagram.py`)

**Purpose:** Clean and filter the combined Instagram JSON into instruction-tuning JSONL records, extracting captions and keywords.

**Input:** `insta-dataset/combined_dataset.json`

**Output:** `instagram_processed.jsonl`

**Command (default arguments):**

```bash
python preprocess_instagram.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` | `instagram_data.json` | Input JSON file path |
| `--output` | `instagram_processed.jsonl` | Output JSONL file path |

> **Note:** When running after Step 2, pass `--input insta-dataset/combined_dataset.json` explicitly:
> ```bash
> python preprocess_instagram.py --input insta-dataset/combined_dataset.json
> ```

---

### Step 4 — YouTube Preprocessing (`preprocess_youtube.py`)

**Purpose:** Clean and filter the raw YouTube CSV into instruction-tuning JSONL records, extracting captions and hashtag keywords from titles and descriptions.

**Input:** `youtube_raw.csv`

**Output:** `youtube_processed.jsonl`

**Command (default arguments):**

```bash
python preprocess_youtube.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--input` | `youtube_raw.csv` | Input CSV file path |
| `--output` | `youtube_processed.jsonl` | Output JSONL file path |

---

### Step 5 — Dataset Merging (`merge_datasets.py`)

**Purpose:** Merge the YouTube and Instagram JSONL files, deduplicate by output text, shuffle, and split into training and validation sets.

**Input:** `youtube_processed.jsonl`, `instagram_processed.jsonl`

**Output:** `train.jsonl`, `val.jsonl`

**Command (default arguments):**

```bash
python merge_datasets.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--youtube` | `youtube_processed.jsonl` | Path to the processed YouTube JSONL file |
| `--instagram` | `instagram_processed.jsonl` | Path to the processed Instagram JSONL file |
| `--train-output` | `train.jsonl` | Output path for the training split |
| `--val-output` | `val.jsonl` | Output path for the validation split |
| `--seed` | `42` | Random seed for shuffling |
| `--train-ratio` | `0.90` | Fraction of records assigned to the training split (must be between 0 and 1) |

---

### Step 6 — Fine-Tuning (`finetune_phi2.py`)

**Purpose:** Fine-tune `microsoft/phi-2` on the merged dataset using QLoRA 4-bit quantization and save the LoRA adapter weights.

**Input:** `train.jsonl`, `val.jsonl`

**Output:** `phi2-caption-finetuned/` (LoRA adapter weights and tokenizer)

**Command (default arguments):**

```bash
python finetune_phi2.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--train-file` | `train.jsonl` | Path to the training JSONL file |
| `--val-file` | `val.jsonl` | Path to the validation JSONL file |
| `--output-dir` | `./phi2-caption-finetuned` | Directory to save the LoRA adapter and tokenizer |
| `--merged-output-dir` | `./phi2-caption-finetuned-merged` | Directory to save the merged base+adapter model |
| `--max-length` | `768` | Maximum token sequence length for training |
| `--num-train-epochs` | `3` | Number of training epochs |
| `--per-device-train-batch-size` | `4` | Per-device training batch size |
| `--gradient-accumulation-steps` | `4` | Number of gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--logging-steps` | `50` | Number of steps between training log entries |
| `--skip-merge` | _(flag, off by default)_ | Skip saving the merged base+adapter weights after training |

---

### Step 7 — Caption Generation (`generate_caption.py`)

**Purpose:** Load the fine-tuned Phi-2 model and generate a platform-formatted caption (30–40 words + 5 hashtags) from a text prompt.

**Input:** `phi2-caption-finetuned/` (LoRA adapter directory or merged model directory), `--prompt` text

**Output:** Formatted caption printed to stdout

**Command (default arguments):**

```bash
python generate_caption.py --prompt "Push through the pain" --model-dir ./phi2-caption-finetuned
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--prompt` | _(required)_ | User prompt text to base the caption on |
| `--model-dir` | _(required)_ | Path to the LoRA adapter directory or merged model directory |
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

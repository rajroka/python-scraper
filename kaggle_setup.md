# Kaggle Phi-2 QLoRA Caption Fine-Tuning

This pipeline fine-tunes `microsoft/phi-2` to generate Instagram-style fitness and motivation captions using the generated `train.jsonl` and `val.jsonl` files.

## 1. Install Dependencies

Run this in a Kaggle notebook cell:

```bash
!pip install -U transformers peft bitsandbytes accelerate datasets trl tqdm
```

Restart the notebook session after installation if Kaggle prompts you to do so.

## 2. Upload Data to Kaggle

1. Run preprocessing locally or in Kaggle:

```bash
python preprocess_youtube.py --input youtube_raw.csv --output youtube_processed.jsonl
python preprocess_instagram.py --input instagram_data.json --output instagram_processed.jsonl
python merge_datasets.py --youtube youtube_processed.jsonl --instagram instagram_processed.jsonl
```

2. Create a Kaggle Dataset containing:

```text
train.jsonl
val.jsonl
finetune_phi2.py
```

3. In the Kaggle notebook, click **Add Input** and attach that dataset.

The files will usually be available under a path like:

```text
/kaggle/input/your-dataset-name/train.jsonl
/kaggle/input/your-dataset-name/val.jsonl
/kaggle/input/your-dataset-name/finetune_phi2.py
```

## 3. GPU Setting

Use a GPU notebook session. Recommended:

- **T4 x2**: preferred for QLoRA training and merging.
- **P100**: usable, but you may need a smaller batch size or `--skip-merge` if memory is tight.

In Kaggle: **Settings -> Accelerator -> GPU T4 x2** or **GPU P100**.

## 4. Run Fine-Tuning

Copy the script and data from the attached dataset into the working directory:

```bash
!cp /kaggle/input/your-dataset-name/finetune_phi2.py /kaggle/working/
!cp /kaggle/input/your-dataset-name/train.jsonl /kaggle/working/
!cp /kaggle/input/your-dataset-name/val.jsonl /kaggle/working/
```

Start training:

```bash
!python /kaggle/working/finetune_phi2.py \
  --train-file /kaggle/working/train.jsonl \
  --val-file /kaggle/working/val.jsonl \
  --output-dir /kaggle/working/phi2-caption-finetuned \
  --merged-output-dir /kaggle/working/phi2-caption-finetuned-merged
```

If the merge step runs out of memory, train only the LoRA adapter:

```bash
!python /kaggle/working/finetune_phi2.py \
  --train-file /kaggle/working/train.jsonl \
  --val-file /kaggle/working/val.jsonl \
  --output-dir /kaggle/working/phi2-caption-finetuned \
  --skip-merge
```

## 5. Download the Fine-Tuned Model

Zip the adapter and merged model directories:

```bash
!zip -r /kaggle/working/phi2-caption-finetuned.zip /kaggle/working/phi2-caption-finetuned
!zip -r /kaggle/working/phi2-caption-finetuned-merged.zip /kaggle/working/phi2-caption-finetuned-merged
```

Then download the `.zip` files from the Kaggle notebook **Output** panel.

The adapter directory is smaller and requires loading the base model plus PEFT adapter. The merged directory is larger and can be loaded directly as a standard causal language model.

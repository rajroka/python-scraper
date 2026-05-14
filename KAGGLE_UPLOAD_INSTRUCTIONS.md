# Complete Kaggle Upload Instructions

The local upload bundle is `kaggle_upload_bundle.zip`. Upload this zip to Kaggle as a Dataset, or upload the files inside it individually.

## Files in the Bundle

- `train.jsonl`
- `val.jsonl`
- `finetune_phi2.py`
- `kaggle_train_phi2.ipynb`
- `kaggle_setup.md`

## Recommended Kaggle Flow

1. Go to Kaggle.
2. Create a new Dataset.
3. Upload `kaggle_upload_bundle.zip`, or upload the files from the `kaggle_upload_bundle` folder.
4. Create a new Notebook.
5. Set accelerator to **GPU T4 x2**. Use **P100** only if T4 x2 is unavailable.
6. Add the Dataset you created as Notebook input.
7. Upload or open `kaggle_train_phi2.ipynb`, then run every cell.
8. Download the generated model zip files from `/kaggle/working`.

## Optional Kaggle CLI Upload

If you want CLI upload from this machine, install Kaggle CLI and place your token at:

```text
C:\Users\lenovo\.kaggle\kaggle.json
```

Then update `kaggle_dataset_metadata.json` by replacing `YOUR_KAGGLE_USERNAME`, copy it into the upload folder as `dataset-metadata.json`, and run:

```bash
kaggle datasets create -p kaggle_upload_bundle
```

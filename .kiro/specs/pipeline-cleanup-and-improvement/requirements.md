# Requirements Document

## Introduction

This feature covers the cleanup and improvement of the fitness/motivation social media caption generation pipeline. The pipeline scrapes YouTube and Instagram data, preprocesses it into instruction-tuning JSONL format, fine-tunes Microsoft Phi-2 with QLoRA 4-bit quantization, and generates captions for Instagram and Facebook posts.

The current implementation has diverged from the Software Requirements Specification (SRS) in several areas: legacy scripts and data files remain alongside the current pipeline, the API key is hardcoded in source code, the inference module described in SRS section 4.6 does not exist, platform-aware output formatting is unimplemented, output format validation is not enforced at inference time, and there is no end-to-end documentation. This feature closes all identified gaps and removes all redundant artefacts.

## Glossary

- **Pipeline**: The end-to-end sequence of scripts that collects, preprocesses, fine-tunes, and generates captions.
- **Preprocessor**: Either `preprocess_instagram.py` or `preprocess_youtube.py` — the current, canonical data-cleaning scripts.
- **Merger**: `merge_datasets.py` — merges, deduplicates, shuffles, and splits JSONL datasets.
- **Fine-Tuner**: `finetune_phi2.py` — QLoRA 4-bit fine-tuning script for Phi-2.
- **Inference_Script**: `generate_caption.py` — the new caption generation script to be created.
- **Output_Validator**: The component (function or module) that checks whether a generated caption meets the 30–40 word and exactly-5-hashtag constraints.
- **Config_Module**: `config.py` — holds scraper configuration including the YouTube Data API key.
- **JSONL_Record**: A single JSON object on one line with keys `instruction`, `input`, and `output`.
- **Legacy_File**: Any file or folder that was part of an earlier pipeline iteration and is no longer referenced by the current pipeline.
- **Platform**: Either `Instagram` or `Facebook` as a target social media channel.
- **Caption**: The body text of a social media post, excluding hashtags, 30–40 words in length.
- **Hashtag_Block**: Exactly 5 hashtags appended to a caption, each prefixed with `#`.
- **Combined_Instagram_JSON**: The file `insta-dataset/combined_dataset.json` that merges all raw `insta-dataset/data-insta-*.json` files, used as input to the Preprocessor.

---

## Requirements

### Requirement 1: Remove Legacy Files and Folders

**User Story:** As a developer, I want all superseded scripts and data files removed from the repository, so that the working directory contains only files that belong to the current pipeline.

#### Acceptance Criteria

1. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `clean_data.py`.
2. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `clean-youtube.py`.
3. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `combine.py`.
4. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `check_data.py`.
5. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `training_data.csv`.
6. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `instarawdata.json`.
7. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `insta-dataset/combined_dataset.json`.
8. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the folder `kaggle_upload_bundle/` nor any of its contents.
9. WHEN cleanup is applied, THE Pipeline SHALL NOT contain the file `kaggle_upload_bundle.zip`.

---

### Requirement 2: Secure API Key Management

**User Story:** As a developer, I want the YouTube Data API key removed from source code, so that credentials are never committed to version control.

#### Acceptance Criteria

1. THE Config_Module SHALL load the YouTube Data API key from the `YOUTUBE_API_KEY` environment variable at runtime.
2. WHEN the Config_Module is loaded, THE Config_Module SHALL return the raw string value of `YOUTUBE_API_KEY` (or an empty string if unset) without raising an exception.
3. WHEN the API key is accessed and the `YOUTUBE_API_KEY` value is absent or empty, THEN THE Config_Module SHALL raise an `EnvironmentError` with a message that names the `YOUTUBE_API_KEY` variable and provides instructions for setting it, and the API call SHALL be aborted.
4. WHEN the API key is accessed and the `YOUTUBE_API_KEY` value contains only whitespace, THEN THE Config_Module SHALL raise an `EnvironmentError` with a message indicating the value is present but invalid (whitespace-only), and the API call SHALL be aborted.
5. THE Config_Module SHALL NOT contain any hardcoded API key string literal.
6. THE Pipeline SHALL include a `.env.example` file that documents the `YOUTUBE_API_KEY` variable name with a placeholder value that is a non-functional string and does not match the YouTube Data API key format (e.g., `YOUTUBE_API_KEY=your-api-key-here`).
7. THE Pipeline SHALL include a `.gitignore` entry that excludes `.env` files from version control.

---

### Requirement 3: Inference Script (Caption Generator)

**User Story:** As a user, I want a script that loads the fine-tuned Phi-2 model and generates a formatted caption from any text prompt, so that I can produce captions without re-running training.

#### Acceptance Criteria

1. THE Inference_Script SHALL accept a user prompt via a `--prompt` command-line argument.
2. THE Inference_Script SHALL accept a `--model-dir` argument pointing to either the LoRA adapter directory or the merged model directory; IF `--model-dir` is not provided or the path does not exist, THEN THE Inference_Script SHALL print an error message to stderr and exit with a non-zero exit code.
3. THE Inference_Script SHALL accept a `--platform` argument with values `instagram` or `facebook`, defaulting to `instagram`; IF an unrecognized value is provided, THEN THE Inference_Script SHALL print an error message to stderr and exit with a non-zero exit code.
4. WHEN the `--platform` argument is `instagram`, THE Inference_Script SHALL prepend `Write a social media caption for Instagram.` to the instruction context.
5. WHEN the `--platform` argument is `facebook`, THE Inference_Script SHALL prepend `Write a social media caption for Facebook.` to the instruction context.
6. WHEN a prompt is provided, THE Inference_Script SHALL produce a response within 5 seconds on hardware meeting the minimum specification (4 GB VRAM GPU).
7. THE Inference_Script SHALL load the model using 4-bit NF4 quantization via `BitsAndBytesConfig` to remain within the 4 GB VRAM constraint; IF model loading fails, THEN THE Inference_Script SHALL print the error to stderr and exit with a non-zero exit code.
8. WHEN the model directory contains LoRA adapter weights (identified by the presence of `adapter_config.json`), THE Inference_Script SHALL load them using `PeftModel` on top of the base `microsoft/phi-2` model.
9. WHEN the model directory does not contain `adapter_config.json`, THE Inference_Script SHALL load the directory directly as a merged causal language model using `AutoModelForCausalLM.from_pretrained`.
10. THE Inference_Script SHALL print the generated caption (30–40 words body followed by exactly 5 hashtags) to stdout.

---

### Requirement 4: Output Format Validation

**User Story:** As a developer, I want every generated caption to be validated against the SRS output constraints before it is shown to the user, so that malformed outputs are caught and retried automatically.

#### Acceptance Criteria

1. THE Output_Validator SHALL count the words in the caption body, where a word is defined as a whitespace-delimited token that does not begin with `#`, and SHALL return `False` if the count is fewer than 30 or greater than 40.
2. THE Output_Validator SHALL count the number of whitespace-delimited tokens beginning with `#` in the full output and SHALL return `False` if the count is not exactly 5.
3. THE Output_Validator SHALL return `True` only when both the word-count and hashtag-count constraints are satisfied simultaneously.
4. WHEN the Output_Validator returns `False`, THE Inference_Script SHALL make up to 3 additional generation attempts (4 total); after all attempts are exhausted, THE Inference_Script SHALL return the output from the final attempt along with a warning message to the caller.
5. THE Output_Validator SHALL be importable as a standalone module so that it can be reused by other pipeline components.

---

### Requirement 5: Platform-Aware Caption Formatting

**User Story:** As a user, I want captions to reflect the stylistic conventions of the target platform, so that Instagram captions and Facebook posts feel appropriate for each channel.

#### Acceptance Criteria

1. WHEN the platform is `instagram`, THE Inference_Script SHALL format the output with the caption body on the first line followed by exactly one newline character and the Hashtag_Block on the next line; IF the formatting step raises an exception, THE Inference_Script SHALL return the unformatted output, defined as the caption body and the Hashtag_Block concatenated with a single space separator.
2. WHEN the platform is `facebook`, THE Inference_Script SHALL format the output with the caption body as a full paragraph followed by exactly two newline characters and the Hashtag_Block; IF the formatting step raises an exception, THE Inference_Script SHALL return the unformatted output, defined as the caption body and the Hashtag_Block concatenated with a single space separator.
3. THE Inference_Script SHALL always print the platform label (`Instagram` or `Facebook`) as the first line of the output, appearing before the caption body, so the user can confirm which style was applied.
4. IF the platform value is neither `instagram` nor `facebook`, THEN THE Inference_Script SHALL return an error message indicating that the platform is unrecognized and no caption output shall be produced.

---

### Requirement 6: Dataset Format Consistency

**User Story:** As a developer, I want all JSONL records in the training and validation datasets to use the `instruction`/`input`/`output` schema, so that the fine-tuning script can load them without format errors.

#### Acceptance Criteria

1. THE Merger SHALL reject any JSONL_Record that is missing the `instruction`, `input`, or `output` key and SHALL log a warning to stderr that includes the source file path and line number.
2. THE Merger SHALL reject any JSONL_Record where the `instruction`, `input`, or `output` value is an empty string or a whitespace-only string (after `.strip()`) and SHALL log a warning to stderr that includes the source file path and line number.
3. THE Merger SHALL write only validated records to the output JSONL files; IF zero valid records remain after filtering, THE Merger SHALL write an empty file and log a warning to stderr indicating that no valid records were found.
4. WHEN the Fine-Tuner successfully loads the dataset, THE Fine-Tuner SHALL proceed directly to tokenization without re-validating field presence.

---

### Requirement 7: Instagram Raw Data Consolidation

**User Story:** As a developer, I want the raw Instagram JSON files consolidated into a single input file before preprocessing, so that the Preprocessor has a single, predictable input path.

#### Acceptance Criteria

1. THE Pipeline SHALL include a `combine_instagram.py` script that reads all `insta-dataset/data-insta-*.json` files and writes the Combined_Instagram_JSON file at `insta-dataset/combined_dataset.json`.
2. WHEN `combine_instagram.py` is run, THE Pipeline SHALL produce a Combined_Instagram_JSON file that contains every record from every source file, deduplicated by the `id` field.
3. WHEN the Preprocessor is invoked with `--input insta-dataset/combined_dataset.json`, THE Preprocessor SHALL exit with code 0 and write at least one record to the output JSONL file.
4. IF a source file in `insta-dataset/` is malformed JSON, THEN `combine_instagram.py` SHALL log an error to stderr that identifies the filename and continue processing the remaining files.
5. IF no files matching `insta-dataset/data-insta-*.json` are found, THEN `combine_instagram.py` SHALL log an error to stderr and exit with a non-zero exit code.

---

### Requirement 8: End-to-End Pipeline Documentation

**User Story:** As a developer, I want a README that documents every step of the pipeline, so that a new contributor can run the full workflow from data collection to caption generation without prior knowledge of the project.

#### Acceptance Criteria

1. THE Pipeline SHALL include a `README.md` file at the repository root.
2. THE `README.md` SHALL include a prerequisites section listing Python 3.10+, minimum hardware (4 GB VRAM GPU, 8 GB RAM), and the exact pip-installable dependencies: `transformers`, `datasets`, `peft`, `bitsandbytes`, `torch`, `accelerate`, `trl`, `tqdm`, `pandas`, `requests`, `langdetect`.
3. THE `README.md` SHALL document each pipeline step in order (YouTube scraping, Instagram data consolidation, Instagram preprocessing, YouTube preprocessing, dataset merging, fine-tuning, caption generation), with each section stating the step's purpose, its input files, and its output files.
4. THE `README.md` SHALL include the exact command to run each pipeline script, showing the default-argument invocation and listing all optional arguments with their default values.
5. THE `README.md` SHALL document the API key setup step, including how to obtain a YouTube Data API key, how to set the `YOUTUBE_API_KEY` environment variable, and how to create a `.env` file from `.env.example`.
6. THE `README.md` SHALL document the expected output files produced by each step: `youtube_raw.csv`, `youtube_processed.jsonl`, `instagram_processed.jsonl`, `train.jsonl`, `val.jsonl`, and the `phi2-caption-finetuned/` adapter weights directory.

# Implementation Plan: Pipeline Cleanup and Improvement

## Overview

This plan converts the design into discrete, incremental coding tasks. Each task builds on the previous ones, ending with full integration. The implementation language is **Python**. Property-based tests use **Hypothesis**; unit tests use **pytest**.

Tasks are ordered to establish foundations first (config, validation utilities), then new scripts, then modifications to existing scripts, then tests, then documentation and cleanup.

---

## Tasks

- [x] 1. Secure API key in `config.py`
  - [x] 1.1 Replace hardcoded `API_KEY` constant with `get_api_key()` function
    - Remove the module-level `API_KEY` string literal from `config.py`
    - Add `import os` at the top of the file
    - Implement `get_api_key() -> str` that reads `os.environ.get("YOUTUBE_API_KEY", "")`, strips the result for the emptiness check only, and raises `EnvironmentError` with a descriptive message (naming the variable and explaining how to set it) if the stripped value is empty
    - Return the raw (unstripped) value when valid
    - _Requirements: 2.1, 2.3, 2.4, 2.5_

  - [x] 1.2 Update `youtube_scraper.py` to call `get_api_key()`
    - Replace `from config import API_KEY` with `from config import get_api_key`
    - Replace every use of `API_KEY` with a call to `get_api_key()` at the point of use (inside `search_videos` and `get_video_description`)
    - _Requirements: 2.1_

  - [x] 1.3 Create `.env.example` and update `.gitignore`
    - Create `.env.example` at the repository root with the content shown in the design (comment block + `YOUTUBE_API_KEY=your-api-key-here`)
    - Create or update `.gitignore` to include `.env`, `*.env`, `__pycache__/`, and `*.pyc` entries
    - _Requirements: 2.6, 2.7_

  - [ ]* 1.4 Write property tests for `config.py`
    - **Property 1: API key round-trip** — for any non-empty, non-whitespace string `s`, setting `YOUTUBE_API_KEY=s` and calling `get_api_key()` returns `s` unchanged
    - **Validates: Requirements 2.1**
    - **Property 2: Whitespace API key raises EnvironmentError** — for any whitespace-only string, `get_api_key()` raises `EnvironmentError`
    - **Validates: Requirements 2.4**
    - Use `@settings(max_examples=100)` on each test
    - Place tests in `tests/test_config.py`

  - [ ]* 1.5 Write unit tests for `config.py`
    - Test: missing env var raises `EnvironmentError`
    - Test: empty string raises `EnvironmentError`
    - Test: importing `config` does not raise even when `YOUTUBE_API_KEY` is unset
    - Place tests in `tests/test_config.py`

- [x] 2. Create `validate_output.py`
  - [x] 2.1 Implement `count_body_words`, `count_hashtags`, and `validate`
    - Create `validate_output.py` at the repository root
    - Implement `count_body_words(text: str) -> int`: split on whitespace, count tokens that do NOT start with `#`
    - Implement `count_hashtags(text: str) -> int`: split on whitespace, count tokens that start with `#`
    - Implement `validate(text: str) -> bool`: return `True` iff `30 <= count_body_words(text) <= 40` and `count_hashtags(text) == 5`
    - No `__main__` block; module is import-only
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [ ]* 2.2 Write property test for `validate_output.py`
    - **Property 3: Output validator correctness** — for any combination of body-word list and hashtag-token list, `validate(joined_text)` equals `(30 <= len(body_words) <= 40) and (len(hashtag_tokens) == 5)`
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - Use `@settings(max_examples=100)` and `st.lists` strategies as shown in the design
    - Place tests in `tests/test_validate_output.py`

  - [ ]* 2.3 Write unit tests for `validate_output.py`
    - Test: empty string returns `False`
    - Test: exactly 30 body words + 5 hashtags returns `True`
    - Test: exactly 40 body words + 5 hashtags returns `True`
    - Test: 29 body words + 5 hashtags returns `False`
    - Test: 41 body words + 5 hashtags returns `False`
    - Test: 35 body words + 4 hashtags returns `False`
    - Test: 35 body words + 6 hashtags returns `False`
    - Place tests in `tests/test_validate_output.py`

- [x] 3. Create `combine_instagram.py`
  - [x] 3.1 Implement `find_source_files`, `load_file`, `deduplicate`, and `main`
    - Create `combine_instagram.py` at the repository root
    - Implement `find_source_files(base_dir: Path) -> list[Path]`: use `base_dir.glob("data-insta-*.json")`, return sorted list
    - Implement `load_file(path: Path) -> list[dict]`: load JSON, return empty list and log to stderr on `json.JSONDecodeError`
    - Implement `deduplicate(records: list[dict]) -> list[dict]`: iterate in order, track seen `id` values in a `set`; keep records without an `id` field unchanged
    - Implement `main()`: parse `--base-dir` (default `insta-dataset`) and `--output` (default `insta-dataset/combined_dataset.json`) args; call `find_source_files`; exit with code 1 to stderr if no files found; load, deduplicate, write output; print summary
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [ ]* 3.2 Write property tests for `combine_instagram.py`
    - **Property 10: Deduplication by id preserves uniqueness** — for any list of records with `id` values drawn from a small integer range, `deduplicate` returns a list where each `id` appears at most once and the first occurrence is retained
    - **Validates: Requirements 7.2**
    - **Property 11: Error-tolerant file processing** — for any mix of valid and malformed JSON files, `combine_instagram.py` processes all valid files, logs errors for malformed ones, and exits 0 when at least one valid file exists
    - **Validates: Requirements 7.4**
    - Use `@settings(max_examples=100)`
    - Place tests in `tests/test_combine_instagram.py`

  - [ ]* 3.3 Write unit tests for `combine_instagram.py`
    - Test: no files found → `sys.exit(1)` is called
    - Test: single valid file → output contains all records from that file
    - Test: two files with overlapping `id` values → duplicates removed, first occurrence kept
    - Test: one valid file + one malformed file → valid records written, error logged to stderr
    - Place tests in `tests/test_combine_instagram.py`

- [x] 4. Modify `merge_datasets.py` — add per-record schema validation
  - [x] 4.1 Implement `validate_record` and integrate into `load_jsonl`
    - Add `validate_record(record: dict, source_path: Path, line_number: int) -> bool` to `merge_datasets.py`
    - Required keys: `instruction`, `input`, `output`; each value must be non-empty after `.strip()`
    - Log warning to stderr in the format `WARNING: Skipping record in <path> line <n>: <reason>` on failure
    - In `load_jsonl`, call `validate_record` after successful JSON parsing; skip records that return `False`
    - After loading both files, if `len(combined) == 0`, log `WARNING: No valid records found after filtering. Output files will be empty.` and write empty output files before exiting with code 0
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 4.2 Write property test for `merge_datasets.py`
    - **Property 9: Merger rejects all invalid records** — for any record missing a required key or having a whitespace-only value, `validate_record` returns `False` and a warning is logged to stderr containing the source path and line number
    - **Validates: Requirements 6.1, 6.2**
    - Use `@settings(max_examples=100)` and `st.fixed_dictionaries` with empty/whitespace/None values
    - Place tests in `tests/test_merge_datasets.py`

  - [ ]* 4.3 Write unit tests for `merge_datasets.py`
    - Test: zero valid records after filtering → empty output files written + warning logged
    - Test: record with all three keys and non-empty values → `validate_record` returns `True`
    - Test: record missing `output` key → `validate_record` returns `False`, warning contains path and line number
    - Test: record with whitespace-only `instruction` → `validate_record` returns `False`
    - Place tests in `tests/test_merge_datasets.py`

- [x] 5. Checkpoint — core utilities complete
  - Ensure all tests pass for tasks 1–4, ask the user if questions arise.

- [ ] 6. Create `generate_caption.py`
  - [ ] 6.1 Implement `parse_args` and `detect_model_type`
    - Create `generate_caption.py` at the repository root
    - Implement `parse_args() -> argparse.Namespace`: define `--prompt` (required), `--model-dir` (required, `Path`), `--platform` (default `instagram`); validate that `--model-dir` exists and `--platform` is one of `instagram`/`facebook`; exit non-zero with stderr message on failure
    - Implement `detect_model_type(model_dir: Path) -> Literal["lora", "merged"]`: return `"lora"` if `adapter_config.json` exists in `model_dir`, else `"merged"`
    - _Requirements: 3.2, 3.3, 3.8, 3.9_

  - [x] 6.2 Implement `load_model_and_tokenizer`
    - Implement `load_model_and_tokenizer(model_dir: Path) -> tuple[PreTrainedModel, PreTrainedTokenizer]`
    - Configure `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16`
    - If `detect_model_type` returns `"lora"`: load base `microsoft/phi-2` with the quantization config, then wrap with `PeftModel.from_pretrained(base_model, model_dir)`
    - If `detect_model_type` returns `"merged"`: load directly with `AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=...)`
    - On any exception, print to stderr and `sys.exit(1)`
    - _Requirements: 3.7, 3.8, 3.9_

  - [x] 6.3 Implement `build_prompt`, `generate_raw`, and `format_caption`
    - Implement `build_prompt(user_prompt: str, platform: str) -> str`: prepend `"Write a social media caption for Instagram."` or `"Write a social media caption for Facebook."` based on platform
    - Implement `generate_raw(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str`: tokenize prompt, call `model.generate`, decode output tokens, return decoded string
    - Implement `format_caption(platform: str, caption_body: str, hashtag_block: str) -> str`: Instagram → `body + "\n" + tags`; Facebook → `body + "\n\n" + tags`; on exception fall back to `body + " " + tags`
    - _Requirements: 3.4, 3.5, 5.1, 5.2_

  - [x] 6.4 Implement `run_with_retry` and `main`
    - Implement `run_with_retry(model, tokenizer, prompt: str, platform: str, max_attempts: int = 4) -> tuple[str, bool]`: loop up to `max_attempts` times calling `generate_raw` then `validate` from `validate_output`; return on first passing result; after exhaustion return last attempt with `passed_validation=False`
    - Implement `main()`: call `parse_args`, `load_model_and_tokenizer`, `build_prompt`, `run_with_retry`; print platform label as first stdout line; print formatted caption; if validation failed print `WARNING: output did not pass validation after 4 attempts` to stderr
    - Import `from validate_output import validate`
    - _Requirements: 3.1, 3.10, 4.4, 5.3, 5.4_

  - [ ]* 6.5 Write property tests for `generate_caption.py`
    - **Property 4: Instagram caption formatting** — for any non-empty `body` and `tags`, `format_caption("instagram", body, tags)` equals `body + "\n" + tags`
    - **Validates: Requirements 5.1**
    - **Property 5: Facebook caption formatting** — for any non-empty `body` and `tags`, `format_caption("facebook", body, tags)` equals `body + "\n\n" + tags`
    - **Validates: Requirements 5.2**
    - **Property 6: Platform label is always the first output line** — for any valid platform and any generated caption, the first line of stdout is the capitalised platform label
    - **Validates: Requirements 5.3**
    - **Property 7: Invalid platform causes non-zero exit** — for any string that is neither `"instagram"` nor `"facebook"`, passing it as `--platform` causes a non-zero exit code
    - **Validates: Requirements 3.3, 5.4**
    - **Property 8: Model loader auto-detects LoRA vs merged** — for any temp directory, presence of `adapter_config.json` causes `detect_model_type` to return `"lora"`; absence causes `"merged"`
    - **Validates: Requirements 3.8, 3.9**
    - Use `@settings(max_examples=100)` on each test
    - Place tests in `tests/test_generate_caption.py`

  - [ ]* 6.6 Write unit tests for `generate_caption.py`
    - Test: missing `--model-dir` → exits non-zero
    - Test: `--model-dir` path does not exist → exits non-zero
    - Test: unrecognised `--platform` value → exits non-zero
    - Test: `run_with_retry` calls the generator exactly 4 times when `validate` always returns `False`
    - Test: `run_with_retry` stops after the first passing attempt and returns `passed_validation=True`
    - Place tests in `tests/test_generate_caption.py`

- [x] 7. Checkpoint — all scripts implemented
  - Ensure all tests pass for tasks 1–6, ask the user if questions arise.

- [x] 8. Create `README.md`
  - [x] 8.1 Write prerequisites, pipeline overview, and API key setup sections
    - Create `README.md` at the repository root
    - Add a prerequisites section: Python 3.10+, minimum hardware (4 GB VRAM GPU, 8 GB RAM), and the exact pip-installable dependencies listed in Requirements 8.2
    - Add an API key setup section: how to obtain a YouTube Data API key, how to set `YOUTUBE_API_KEY`, and how to copy `.env.example` to `.env`
    - _Requirements: 8.1, 8.2, 8.5_

  - [x] 8.2 Document each pipeline step with commands and file I/O
    - Add one section per pipeline step in order: YouTube scraping, Instagram data consolidation, Instagram preprocessing, YouTube preprocessing, dataset merging, fine-tuning, caption generation
    - Each section states the step's purpose, input files, output files, and the exact command to run the script (default-argument invocation + all optional arguments with defaults)
    - Document expected output files: `youtube_raw.csv`, `youtube_processed.jsonl`, `instagram_processed.jsonl`, `train.jsonl`, `val.jsonl`, `phi2-caption-finetuned/`
    - _Requirements: 8.3, 8.4, 8.6_

- [x] 9. Remove legacy files
  - [x] 9.1 Delete legacy scripts and data files
    - Delete `clean_data.py`
    - Delete `clean-youtube.py`
    - Delete `combine.py`
    - Delete `check_data.py`
    - Delete `training_data.csv`
    - Delete `instarawdata.json`
    - Delete `insta-dataset/combined_dataset.json`
    - Recursively delete `kaggle_upload_bundle/` directory and all its contents
    - Delete `kaggle_upload_bundle.zip`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

- [x] 10. Final checkpoint — full pipeline clean
  - Ensure all tests pass, no legacy files remain, and the README accurately reflects the current state of the repository. Ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 5, 7, and 10 ensure incremental validation
- Property tests (Hypothesis) validate universal correctness properties across 100 generated examples each
- Unit tests validate specific examples, edge cases, and error conditions
- Task 9 (legacy file deletion) is placed last to avoid breaking any test that might reference those files during development
- The `tests/` directory should be created as part of task 1.4 (first test file)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.2", "2.3", "3.1"] },
    { "id": 2, "tasks": ["1.4", "1.5", "3.2", "3.3", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "6.1"] },
    { "id": 4, "tasks": ["6.2"] },
    { "id": 5, "tasks": ["6.3"] },
    { "id": 6, "tasks": ["6.4"] },
    { "id": 7, "tasks": ["6.5", "6.6", "8.1"] },
    { "id": 8, "tasks": ["8.2", "9.1"] }
  ]
}
```

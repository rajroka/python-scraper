#!/usr/bin/env python3
"""Merge, deduplicate, shuffle, and split caption fine-tuning JSONL datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge processed YouTube and Instagram JSONL files.")
    parser.add_argument("--youtube", type=Path, default=Path("youtube_ready.jsonl"))
    parser.add_argument("--instagram", type=Path, default=Path("instagram_ready.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--val-output", type=Path, default=Path("val.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    return parser.parse_args()


REQUIRED_KEYS = ("instruction", "input", "output")


def validate_record(record: dict, source_path: Path, line_number: int) -> bool:
    """Return True if record has all required keys with non-whitespace values.

    Logs a warning to stderr with source_path and line_number on failure.
    Required keys: 'instruction', 'input', 'output'.
    """
    for key in REQUIRED_KEYS:
        if key not in record:
            print(
                f"WARNING: Skipping record in {source_path} line {line_number}: missing key '{key}'",
                file=sys.stderr,
            )
            return False
        if not str(record[key]).strip():
            print(
                f"WARNING: Skipping record in {source_path} line {line_number}: empty value for key '{key}'",
                file=sys.stderr,
            )
            return False
    return True


def load_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(tqdm(jsonl_file, desc=f"Loading {path.name}"), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"WARNING: Skipping invalid JSON in {path} line {line_number}: {exc}")
                    continue
                if isinstance(record, dict) and validate_record(record, path, line_number):
                    records.append(record)
    except FileNotFoundError:
        print(f"ERROR: Could not find input file: {path.resolve()}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to load JSONL file {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)
    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    try:
        with path.open("w", encoding="utf-8") as jsonl_file:
            for record in tqdm(records, desc=f"Writing {path.name}"):
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"ERROR: Failed to write JSONL file {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        print("ERROR: --train-ratio must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    print("Loading processed datasets")
    youtube_records = load_jsonl(args.youtube)
    instagram_records = load_jsonl(args.instagram)
    combined = youtube_records + instagram_records
    print(f"Loaded {len(youtube_records)} YouTube records")
    print(f"Loaded {len(instagram_records)} Instagram records")

    if len(combined) == 0:
        print(
            "WARNING: No valid records found after filtering. Output files will be empty.",
            file=sys.stderr,
        )
        write_jsonl(args.train_output, [])
        write_jsonl(args.val_output, [])
        sys.exit(0)

    deduped: list[dict[str, str]] = []
    seen_outputs: set[str] = set()
    for record in tqdm(combined, desc="Deduplicating by output"):
        output = str(record.get("output", "")).strip()
        if not output or output in seen_outputs:
            continue
        seen_outputs.add(output)
        deduped.append(record)

    random.seed(args.seed)
    random.shuffle(deduped)

    split_index = int(len(deduped) * args.train_ratio)
    train_records = deduped[:split_index]
    val_records = deduped[split_index:]

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.val_output, val_records)

    print(f"Combined records before dedupe: {len(combined)}")
    print(f"Records after dedupe: {len(deduped)}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")


if __name__ == "__main__":
    main()

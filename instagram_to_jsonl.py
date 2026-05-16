#!/usr/bin/env python3
"""instagram_to_jsonl.py — Convert insta_clean.json into instruction-tuning JSONL.

Reads the reviewed insta_clean.json (produced by clean_instagram.py) and writes
each record as a single JSON line to instagram_ready.jsonl.

Run clean_instagram.py first, review/edit insta_clean.json, then run this script.

Usage:
    python instagram_to_jsonl.py [--input insta_clean.json] [--output instagram_ready.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert insta_clean.json into instagram_ready.jsonl."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("insta_clean.json"),
        help="Input clean JSON path (default: insta_clean.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("instagram_ready.jsonl"),
        help="Output JSONL path (default: instagram_ready.jsonl)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find input file: {path.resolve()}", file=sys.stderr)
        print("Run clean_instagram.py first to generate insta_clean.json.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to load {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(
            f"ERROR: {path} must be a JSON array of records. Re-run clean_instagram.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    return [item for item in data if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"instruction", "input", "output"}


def validate_record(record: dict[str, Any]) -> bool:
    """Return True if the record has all required keys with non-empty string values."""
    for key in REQUIRED_KEYS:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    print(f"Loading clean Instagram JSON: {args.input}")
    records = load_records(args.input)
    print(f"Loaded {len(records)} records")

    valid = 0
    skipped = 0

    try:
        with args.output.open("w", encoding="utf-8") as out_file:
            for record in tqdm(records, desc="Writing JSONL"):
                if not validate_record(record):
                    skipped += 1
                    continue
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                valid += 1
    except Exception as exc:
        print(f"ERROR: Failed to write {args.output.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Records written: {valid}")
    print(f"Records skipped: {skipped}")
    print(f"Output:          {args.output}")


if __name__ == "__main__":
    main()

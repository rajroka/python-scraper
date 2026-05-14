#!/usr/bin/env python3
"""Preprocess Apify Instagram scraper JSON into instruction-tuning JSONL records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


INSTRUCTION = "Write an engaging Instagram caption for a fitness and motivation post."
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Instagram JSON into JSONL fine-tuning data.")
    parser.add_argument("--input", type=Path, default=Path("instagram_data.json"), help="Input JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("instagram_processed.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find input file: {path.resolve()}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to load JSON file {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("items", "data", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    print("ERROR: Instagram JSON must be a list of records or contain an items/data/results list.", file=sys.stderr)
    sys.exit(1)


def normalize_newlines(caption: str) -> str:
    caption = repair_mojibake(caption)
    caption = caption.replace("\r\n", "\n").replace("\r", "\n")
    caption = re.sub(r"[ \t]+\n", "\n", caption)
    caption = re.sub(r"\n[ \t]+", "\n", caption)
    caption = re.sub(r"\n{3,}", "\n\n", caption)
    return caption.strip()


def repair_mojibake(text: str) -> str:
    if not text:
        return ""
    if any(marker in text for marker in ("Ã", "Â", "â", "ðŸ")):
        try:
            repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
            if len(repaired.strip()) >= len(text.strip()) * 0.75:
                return repaired
        except UnicodeError:
            return text
    return text


def hashtag_ratio(text: str) -> float:
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0
    hashtag_words = sum(1 for word in words if word.startswith("#"))
    return hashtag_words / len(words)


def extract_niche(input_url: str) -> str:
    if not input_url:
        return ""
    path_parts = [part for part in urlparse(input_url).path.split("/") if part]
    if "tags" in path_parts:
        tag_index = path_parts.index("tags")
        if tag_index + 1 < len(path_parts):
            return path_parts[tag_index + 1].strip().lower()
    return path_parts[-1].strip().lower() if path_parts else ""


def normalize_hashtag(value: Any) -> str:
    tag = str(value).strip().lstrip("#").lower()
    return tag


def extract_keywords(record: dict[str, Any], niche: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()

    hashtags = record.get("hashtags") or []
    if isinstance(hashtags, str):
        hashtags = HASHTAG_RE.findall(hashtags) or [part.strip() for part in hashtags.split(",")]

    if isinstance(hashtags, list):
        for raw_tag in hashtags:
            keyword = normalize_hashtag(raw_tag)
            if not keyword or keyword in seen:
                continue
            seen.add(keyword)
            keywords.append(keyword)
            if len(keywords) == 5:
                return keywords

    if niche and niche not in seen:
        keywords.append(niche)
    return keywords[:5]


def make_record(output_text: str, keywords: list[str]) -> dict[str, str]:
    keyword_text = ", ".join(keywords) if keywords else "fitness, motivation, gym"
    return {
        "instruction": INSTRUCTION,
        "input": f"Keywords: {keyword_text}",
        "output": output_text,
    }


def main() -> None:
    args = parse_args()
    print(f"Loading Instagram JSON: {args.input}")
    records = load_records(args.input)
    print(f"Loaded {len(records)} records")

    valid = 0
    skipped = 0

    try:
        with args.output.open("w", encoding="utf-8") as out_file:
            for record in tqdm(records, desc="Processing Instagram records"):
                raw_caption = record.get("caption")
                if not isinstance(raw_caption, str) or not raw_caption.strip():
                    skipped += 1
                    continue

                caption = normalize_newlines(raw_caption)
                if len(caption) < 50 or len(caption) > 2200:
                    skipped += 1
                    continue
                if hashtag_ratio(caption) > 0.60:
                    skipped += 1
                    continue

                niche = extract_niche(str(record.get("inputUrl", "") or ""))
                keywords = extract_keywords(record, niche)
                out_file.write(json.dumps(make_record(caption, keywords), ensure_ascii=False) + "\n")
                valid += 1
    except Exception as exc:
        print(f"ERROR: Failed to write JSONL file {args.output.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved cleaned Instagram data to: {args.output}")
    print(f"Total valid records: {valid}")
    print(f"Total skipped records: {skipped}")


if __name__ == "__main__":
    main()

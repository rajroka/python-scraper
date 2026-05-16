#!/usr/bin/env python3
"""Preprocess scraped YouTube data into instruction-tuning JSONL records."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the script usable if tqdm is absent.
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


INSTRUCTION = "Write an engaging Instagram caption for a fitness and motivation post."

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
TIMESTAMP_RE = re.compile(r"^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\s*[-–—].*)?\s*$")
TRACK_RE = re.compile(r"^\s*[^-\n]{2,80}\s[-–—]\s[^-\n]{2,120}\s*$")
PIPE_SUFFIX_RE = re.compile(r"\s*\|\s*[^|]{2,80}$")
LEGAL_RE = re.compile(
    r"\b("
    r"copyright|all rights reserved|fair use|disclaimer|no copyright|"
    r"rights belong|owned by|trademark|terms of use|privacy policy|"
    r"affiliate|sponsored|for business inquiries|business inquiry|"
    r"contact us|dm for credit|credit to|do not own|declaration|"
    r"do not attempt|serious injury|paralysis|death"
    r")\b",
    re.IGNORECASE,
)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean YouTube rows into JSONL fine-tuning data.")
    parser.add_argument("--input", type=Path, default=Path("youtube_raw.csv"), help="Input CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("youtube_processed.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except FileNotFoundError:
        print(f"ERROR: Could not find input file: {path.resolve()}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to load CSV file {path.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def clean_title(title: str) -> str:
    title = repair_mojibake(title or "")
    title = PIPE_SUFFIX_RE.sub("", title)
    title = HASHTAG_RE.sub("", title)
    return normalize_spaces(title)


def hashtag_ratio(line: str) -> float:
    words = re.findall(r"\S+", line)
    if not words:
        return 0.0
    hashtag_words = sum(1 for word in words if word.startswith("#"))
    return hashtag_words / len(words)


def is_bad_description_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 20:
        return True
    if TIMESTAMP_RE.match(stripped):
        return True
    if URL_RE.search(stripped):
        return True
    if EMAIL_RE.search(stripped):
        return True
    if LEGAL_RE.search(stripped):
        return True
    if hashtag_ratio(stripped) > 0.60:
        return True
    if TRACK_RE.match(stripped):
        return True
    return False


def clean_description(description: str) -> str:
    kept_lines: list[str] = []
    for raw_line in repair_mojibake(description or "").splitlines():
        line = normalize_spaces(raw_line)
        if not line or is_bad_description_line(line):
            continue
        kept_lines.append(line)

    cleaned = normalize_spaces(" ".join(kept_lines))
    return cleaned[:500].rstrip()


def extract_keywords(*texts: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in HASHTAG_RE.findall(text or ""):
            keyword = match.strip("_").lower()
            if not keyword or keyword in seen:
                continue
            seen.add(keyword)
            keywords.append(keyword)
            if len(keywords) == 5:
                return keywords
    return keywords


def make_record(output_text: str, keywords: list[str]) -> dict[str, str]:
    keyword_text = ", ".join(keywords) if keywords else "fitness, motivation, gym"
    return {
        "instruction": INSTRUCTION,
        "input": f"Keywords: {keyword_text}",
        "output": output_text,
    }


def main() -> None:
    args = parse_args()
    print(f"Loading YouTube CSV: {args.input}")
    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows")

    valid = 0
    skipped = 0

    try:
        with args.output.open("w", encoding="utf-8") as out_file:
            for row in tqdm(rows, desc="Processing YouTube rows"):
                title = row.get("title", "") or ""
                description = row.get("description", "") or ""
                keywords = extract_keywords(title, description)

                title_caption = clean_title(title)
                description_caption = clean_description(description)

                candidates = [title_caption, description_caption]
                row_valid = False
                for caption in candidates:
                    if URL_RE.search(caption):
                        continue
                    if caption == title_caption and len(caption) <= 20:
                        continue
                    if caption == description_caption and len(caption) <= 80:
                        continue

                    out_file.write(json.dumps(make_record(caption, keywords), ensure_ascii=False) + "\n")
                    valid += 1
                    row_valid = True

                if not row_valid:
                    skipped += 1
    except Exception as exc:
        print(f"ERROR: Failed to write JSONL file {args.output.resolve()}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved cleaned YouTube data to: {args.output}")
    print(f"Total valid records: {valid}")
    print(f"Total skipped rows: {skipped}")


if __name__ == "__main__":
    main()

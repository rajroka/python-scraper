#!/usr/bin/env python3
"""
STRICT YouTube → Instagram Caption Dataset Cleaner
For PostSathi fine-tuning (HIGH QUALITY ONLY)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

# ─────────────────────────────
# CONFIG
# ─────────────────────────────

INSTRUCTION = "Write an engaging Instagram fitness and motivation caption with exactly 5 hashtags."

MIN_WORDS = 30
MAX_WORDS = 40
REQUIRED_HASHTAGS = 5

# ─────────────────────────────
# REGEX
# ─────────────────────────────

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
LEGAL_RE = re.compile(
    r"copyright|all rights reserved|subscribe|like|follow|"
    r"contact|dm|promo|affiliate|disclaimer|terms|privacy",
    re.IGNORECASE
)

BAD_CHARS_RE = re.compile(r"[^\w\s#,.!?]")  # removes messy symbols

# ─────────────────────────────
# LOAD CSV
# ─────────────────────────────

def load_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print("CSV ERROR:", e)
        sys.exit(1)

# ─────────────────────────────
# CLEANING
# ─────────────────────────────

def clean_text(text: str) -> str:
    text = text or ""
    text = text.strip()

    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = BAD_CHARS_RE.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def word_count(text: str) -> int:
    return len(text.split())


def extract_hashtags(text: str) -> list[str]:
    tags = HASHTAG_RE.findall(text)
    clean_tags = []
    seen = set()

    for t in tags:
        t = t.lower().strip()
        if t and t not in seen:
            seen.add(t)
            clean_tags.append(t)
        if len(clean_tags) == REQUIRED_HASHTAGS:
            break

    return clean_tags


def is_valid(text: str, tags: list[str]) -> bool:
    if not text:
        return False

    if LEGAL_RE.search(text):
        return False

    wc = word_count(text)

    if wc < MIN_WORDS or wc > MAX_WORDS:
        return False

    if len(tags) != REQUIRED_HASHTAGS:
        return False

    if URL_RE.search(text) or EMAIL_RE.search(text):
        return False

    return True


# ─────────────────────────────
# BUILD CAPTION
# ─────────────────────────────

def build_caption(text: str, tags: list[str]) -> str:
    tag_str = " ".join(f"#{t}" for t in tags)

    return f"{text}\n\n{tag_str}"


def extract_keywords(title: str, desc: str) -> list[str]:
    return list(dict.fromkeys(
        HASHTAG_RE.findall(title + " " + desc)
    ))[:5]


# ─────────────────────────────
# MAIN
# ─────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="youtube_raw.csv")
    parser.add_argument("--output", default="youtube_clean.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))

    output = []
    seen = set()
    skipped = 0

    for r in rows:
        title = clean_text(r.get("title", ""))
        desc = clean_text(r.get("description", ""))

        base = desc if len(desc) > len(title) else title

        tags = extract_keywords(title, desc)

        if len(tags) < REQUIRED_HASHTAGS:
            skipped += 1
            continue

        caption = base

        if not is_valid(caption, tags):
            skipped += 1
            continue

        caption = build_caption(caption, tags)

        if word_count(caption.split("\n")[0]) < MIN_WORDS:
            skipped += 1
            continue

        key = caption.lower().strip()

        if key in seen:
            continue

        seen.add(key)

        output.append({
            "instruction": INSTRUCTION,
            "input": f"Keywords: {', '.join(tags)}",
            "output": caption
        })

    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("DONE")
    print("Saved:", args.output)
    print("Total:", len(output))
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()
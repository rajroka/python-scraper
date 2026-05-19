#!/usr/bin/env python3

# =========================================================
# STRICT INSTAGRAM DATASET CLEANER
# POSTSATHI DATASET PIPELINE
#
# Run:
# python strict_clean_instagram.py
#
# Optional:
# python strict_clean_instagram.py \
#   --input insta-dataset/combined_dataset.json \
#   --output insta_clean_strict.json
#
# Install:
# pip install tqdm rapidfuzz
# =========================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from rapidfuzz.fuzz import ratio

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# =========================================================
# CONFIG
# =========================================================

INSTRUCTION = (
    "Write an engaging Instagram caption for a fitness and motivation post."
)

MIN_WORDS = 40
MAX_WORDS = 120

MIN_SENTENCES = 2
MAX_HASHTAGS = 5

SIMILARITY_THRESHOLD = 88

ALLOWED_TOPICS = [
    "fitness",
    "gym",
    "motivation",
    "mindset",
    "discipline",
    "success",
    "workout",
    "focus",
    "consistency",
    "transformation",
    "selfgrowth",
    "selfimprovement",
    "growth",
    "confidence",
]

SPAM_WORDS = [
    "buy now",
    "dm me",
    "link in bio",
    "discount",
    "sale",
    "promo",
    "giveaway",
    "shop now",
    "subscribe",
    "follow for follow",
    "affiliate",
    "amazon",
    "bitcoin",
    "crypto signal",
]

PROMO_PATTERNS = [
    r"\+\d{1,3}",
    r"\bcall\b",
    r"\bcontact\b",
    r"\bvisit\b",
    r"\bbook now\b",
    r"\bjoin now\b",
    r"\bavailable now\b",
    r"\bshop\b",
    r"\.com\b",
    r"\.net\b",
    r"\.io\b",
]

GENERIC_PHRASES = [
    "never quit",
    "stay focused",
    "trust the process",
    "stay disciplined",
    "hard work pays off",
    "consistency is key",
    "keep pushing",
]

CTA_WORDS = [
    "comment",
    "share",
    "follow",
    "tag someone",
    "dm me",
]

AI_PATTERNS = [
    "the goal is simple",
    "become better than yesterday",
    "consistency > perfection",
    "results come from consistency",
]

HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("insta-dataset/combined_dataset.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("insta_clean_strict.json"),
    )

    return parser.parse_args()


# =========================================================
# LOAD DATA
# =========================================================

def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as exc:
        print(f"ERROR loading JSON: {exc}")
        sys.exit(1)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("items", "data", "results"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]

    print("ERROR: Invalid JSON structure")
    sys.exit(1)


# =========================================================
# TEXT CLEANING
# =========================================================

def repair_mojibake(text: str) -> str:
    if not text:
        return ""

    if any(x in text for x in ("Ã", "Â", "â", "ðŸ")):
        try:
            return text.encode("cp1252").decode("utf-8")
        except Exception:
            return text

    return text


def normalize_text(text: str) -> str:
    text = repair_mojibake(text)

    text = text.replace("\r", "\n")

    text = re.sub(r"\n+", "\n", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_urls(text: str) -> str:
    return URL_RE.sub("", text)


def remove_hashtags(text: str) -> str:
    return HASHTAG_RE.sub("", text).strip()


def remove_extra_emojis(text: str) -> str:
    emojis = EMOJI_RE.findall(text)

    if len(emojis) > 8:
        text = EMOJI_RE.sub("", text)

    return text


# =========================================================
# QUALITY CHECKS
# =========================================================

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]+", text))


def extract_hashtags(text: str) -> list[str]:
    seen = set()
    tags = []

    for tag in HASHTAG_RE.findall(text):
        tag = tag.lower().strip()

        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    return tags


def contains_spam(text: str) -> bool:
    lower = text.lower()

    for spam in SPAM_WORDS:
        if spam in lower:
            return True

    return False


def contains_promo(text: str) -> bool:
    lower = text.lower()

    for pattern in PROMO_PATTERNS:
        if re.search(pattern, lower):
            return True

    return False


def too_generic(text: str) -> bool:
    lower = text.lower()

    matches = sum(
        phrase in lower
        for phrase in GENERIC_PHRASES
    )

    return matches >= 2


def excessive_cta(text: str) -> bool:
    lower = text.lower()

    count = sum(
        word in lower
        for word in CTA_WORDS
    )

    return count >= 2


def repetitive_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())

    if not words:
        return 1.0

    unique = len(set(words))

    return unique / len(words)


def emoji_ratio(text: str) -> float:
    emojis = EMOJI_RE.findall(text)

    if not text:
        return 0

    return len("".join(emojis)) / len(text)


def is_english_like(text: str) -> bool:
    stripped = EMOJI_RE.sub("", text)

    if not stripped:
        return False

    english_chars = sum(c.isascii() for c in stripped)

    ratio_ = english_chars / len(stripped)

    return ratio_ > 0.90


def topic_relevant(text: str) -> bool:
    lower = text.lower()

    matches = sum(
        1
        for topic in ALLOWED_TOPICS
        if topic in lower
    )

    return matches >= 2


def ai_pattern_detected(text: str) -> bool:
    lower = text.lower()

    matches = sum(
        pattern in lower
        for pattern in AI_PATTERNS
    )

    return matches >= 2


# =========================================================
# CLEAN CAPTION
# =========================================================

def clean_caption(caption: str) -> str | None:

    caption = normalize_text(caption)

    if len(extract_hashtags(caption)) > 8:
        return None

    caption = remove_urls(caption)

    caption = remove_extra_emojis(caption)

    caption = remove_hashtags(caption)

    if contains_spam(caption):
        return None

    if contains_promo(caption):
        return None

    if excessive_cta(caption):
        return None

    if too_generic(caption):
        return None

    if ai_pattern_detected(caption):
        return None

    if not is_english_like(caption):
        return None

    if not topic_relevant(caption):
        return None

    if repetitive_ratio(caption) < 0.58:
        return None

    if emoji_ratio(caption) > 0.12:
        return None

    if sentence_count(caption) < MIN_SENTENCES:
        return None

    words = word_count(caption)

    if words < MIN_WORDS or words > MAX_WORDS:
        return None

    return caption.strip()


# =========================================================
# KEYWORDS
# =========================================================

def extract_niche(input_url: str) -> str:

    if not input_url or not input_url.startswith("http"):
        return ""

    path_parts = [
        p for p in urlparse(input_url).path.split("/")
        if p
    ]

    if "tags" in path_parts:
        idx = path_parts.index("tags")

        if idx + 1 < len(path_parts):
            return path_parts[idx + 1].lower()

    return ""


def extract_keywords(
    record: dict[str, Any],
    niche: str,
) -> list[str]:

    keywords = []
    seen = set()

    hashtags = record.get("hashtags") or []

    if isinstance(hashtags, str):
        hashtags = HASHTAG_RE.findall(hashtags)

    if isinstance(hashtags, list):

        for tag in hashtags:

            tag = str(tag).lower()
            tag = tag.replace("#", "").strip()

            if not tag:
                continue

            if tag in seen:
                continue

            if len(tag) < 3:
                continue

            seen.add(tag)

            keywords.append(tag)

            if len(keywords) == MAX_HASHTAGS:
                break

    if niche and niche not in seen:
        keywords.append(niche)

    return keywords[:MAX_HASHTAGS]


# =========================================================
# FORMAT OUTPUT
# =========================================================

def build_output(
    caption: str,
    hashtags: list[str],
) -> str:

    hashtags = hashtags[:MAX_HASHTAGS]

    formatted_tags = " ".join(
        f"#{x}"
        for x in hashtags
    )

    return f"{caption}\n\n{formatted_tags}"


def make_record(
    caption: str,
    keywords: list[str],
) -> dict[str, str]:

    return {
        "instruction": INSTRUCTION,
        "input": f"Keywords: {', '.join(keywords)}",
        "output": build_output(caption, keywords),
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    args = parse_args()

    print(f"Loading: {args.input}")

    records = load_records(args.input)

    print(f"Loaded {len(records)} records")

    clean_records = []

    skipped = 0

    seen_outputs = []

    for record in tqdm(records):

        raw_caption = record.get("caption")

        if not isinstance(raw_caption, str):
            skipped += 1
            continue

        cleaned = clean_caption(raw_caption)

        if not cleaned:
            skipped += 1
            continue

        # =========================================
        # FUZZY DUPLICATE DETECTION
        # =========================================

        is_duplicate = any(
            ratio(cleaned, existing) > SIMILARITY_THRESHOLD
            for existing in seen_outputs
        )

        if is_duplicate:
            skipped += 1
            continue

        seen_outputs.append(cleaned)

        niche = extract_niche(
            str(record.get("inputUrl", ""))
        )

        keywords = extract_keywords(
            record,
            niche,
        )

        if len(keywords) < 3:
            keywords.extend([
                "fitness",
                "motivation",
                "mindset",
            ])

        keywords = keywords[:MAX_HASHTAGS]

        clean_records.append(
            make_record(
                cleaned,
                keywords,
            )
        )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean_records,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=================================")
    print(f"Records written: {len(clean_records)}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {args.output}")
    print("=================================")


if __name__ == "__main__":
    main()
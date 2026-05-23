#!/usr/bin/env python3

# =========================================================
# INSTAGRAM DATASET CLEANER
# POSTSATHI DATASET PIPELINE
#
# Run:
# python clean_instagram.py
#
# Optional:
# python clean_instagram.py \
#   --input insta-dataset/combined_dataset.json \
#   --output insta_clean.json
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

MIN_WORDS         = 40
MAX_WORDS         = 60
MIN_SENTENCES     = 2
MAX_HASHTAGS      = 5
MAX_AT_MENTIONS   = 2          # posts with 3+ @mentions → promo/sponsor-heavy
MAX_CAMELCASE     = 3          # inline CamelCase word dumps (shadowhashtags)
TITLE_CASE_RATIO  = 0.55       # >55% title-cased words → Every Word Like This
MIN_TITLE_WORDS   = 12         # only apply title-case check on longer text

SIMILARITY_THRESHOLD = 88

ALLOWED_TOPICS = [
    "fitness", "gym", "motivation", "mindset", "discipline",
    "success", "workout", "focus", "consistency", "transformation",
    "selfgrowth", "selfimprovement", "growth", "confidence",
]

SPAM_WORDS = [
    # ---- Original ----
    "buy now", "dm me", "link in bio", "discount", "sale", "promo",
    "giveaway", "shop now", "subscribe", "follow for follow",
    "affiliate", "amazon", "bitcoin", "crypto signal",

    # ---- Supplements / products ----
    "supplement", "protein powder", "pre-workout", "pre workout",
    "mass gainer", "amino acid", "fat burner", "whey protein",
    "creatine", "intra-workout", "post-workout",

    # ---- Swipe CTAs ----
    "swipe left", "swipe right", "swipe to see", "swipe up",

    # ---- Trainer / gym service pitches ----
    "opening spots", "spots available", "limited spots",
    "1:1 coaching", "1:1 personal training", "personal training",
    "nutrition guidance", "nutrition coaching",
    "weekly accountability", "accountability check",
    "message me if", "dm if you", "dm for",
    "enrollment", "enroll now",
    "bring a friend", "free trial", "first class free",
    "no sign-up fee", "no signup fee",
    "join us",

    # ---- Business / financial ----
    "tax planning", "tax advisor", "tax efficiency", "tax clarity",
    "taxplanning", "taxadvisor",
    "business coaching", "financial planning", "invoice",
    "revenue", "roi", "profit",
    "wealth creation", "millionaire", "passive income",
    "wealth mindset", "rich mindset",
    "financial freedom", "financial growth",

    # ---- Location / store promo ----
    "best value", "open 24/7", "open 24 7",
    "find us at", "located at", "visit our", "visit us at",
    "now available at", "in store",

    # ---- Competition / event announcements ----
    "turning pro", "turned pro", "turning professional",
    "competition day", "show day", "stage ready",
    "prejudging", "peak week",
    "title secured", "first place",
    "award night", "prize money",

    # ---- App promotion ----
    "download the app", "app store", "google play",
    "our app", "the app",

    # ---- Religious figure promotion ----
    "saint dr", "ji insan", "gurmeet", "ram rahim",
    "teaches that",
]

PROMO_PATTERNS = [
    # ---- Original ----
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

    # ---- Pricing / deals ----
    r"\$\d+",
    r"\bfree\b",
    r"\bprice[ds]?\b",
    r"\bpricing\b",
    r"\bpackage[sd]?\b",
    r"\boffer[sd]?\b",

    # ---- Location signals ----
    r"📍",
    r"\baddress\b",
    r"\bopen\s+\d",

    # ---- Street addresses ----
    r"\b\d{3,5}\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Dr|Lane|Street|Avenue|Road)\b",

    # ---- Spot / slot availability ----
    r"\d+\s*spots?\b",
    r"\bspots?\s+available\b",
    r"\bslots?\s+available\b",

    # ---- Hours / schedules ----
    r"\bhours?\s*:",
    r"\bam\b.{0,10}\bpm\b",

    # ---- Day + time = class schedule ----
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b.{0,60}\d{1,2}[:.]\d{2}",

    # ---- Product / course launch ----
    r"\blaunching\b",
    r"\bcoming\s+soon\b",
    r"\bdrops?\s+(soon|\w+\s+\d+)\b",
    r"\blaunch\s+date\b",
    r"\bpre.?order\b",

    # ---- App CTAs ----
    r"\bdownload\b",
    r"\bapp\s+store\b",
    r"\bgoogle\s+play\b",
    r"\bin\s+(?:the\s+)?\w+\s+app\b",

    # ---- Product listing emojis ----
    r"⚡\s*\w+",
    r"✨\s*\w+",
    r"►\s*\w+",

    # ---- Marketing copy language ----
    r"\bideal\s+solution\b",
    r"\bhighly\s+effective\b",
    r"\ball\s+fitness\s+levels\b",
    r"\bpresents\s+an\s+ideal\b",
    r"\baccommodates\b",
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
    "comment", "share", "follow", "tag someone", "dm me",
]

AI_PATTERNS = [
    "the goal is simple",
    "become better than yesterday",
    "consistency > perfection",
    "results come from consistency",
]

# Words that clearly signal non-English text.
# Selected to avoid overlap with English words.
NON_ENGLISH_INDICATORS = {
    # Norwegian / Danish / Swedish
    "ikke", "utvikling", "fremover", "noen", "ganger",
    "trening", "disiplin", "fremtiden", "fortsette",
    # Spanish
    "pero", "también", "porque", "cuando", "siempre", "nunca",
    "ellos", "nuestro",
    # French
    "avec", "nous", "vous", "toujours", "jamais", "notre",
    "dans",
    # German
    "nicht", "auch", "aber", "oder", "wenn", "schon",
    "sehr", "wird", "haben", "werden",
    # Italian
    "anche", "perché", "molto", "della", "dello",
    # Dutch
    "niet", "maar", "wordt", "zijn",
    # Portuguese
    "também", "quando", "porque",
}

# English function words for language verification
ENGLISH_FUNCTION_WORDS = {
    "the", "is", "are", "was", "were", "you", "your", "we", "our",
    "in", "of", "and", "a", "to", "it", "that", "this", "with",
    "for", "on", "not", "be", "have", "has", "do", "does", "but",
    "from", "at", "by", "an", "or", "if", "can", "will", "just",
    "when", "what", "how", "who", "all", "so", "up", "about",
    "they", "their", "them", "he", "she", "his", "her", "its",
    "been", "had", "would", "could", "should", "than", "more",
}

HASHTAG_RE      = re.compile(r"#([A-Za-z0-9_]+)")
AT_MENTION_RE   = re.compile(r"@[A-Za-z0-9_.]+")
URL_RE          = re.compile(r"https?://\S+|www\.\S+")
CAMELCASE_RE    = re.compile(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b")

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
        default=Path("insta_clean.json"),
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


def remove_at_mentions(text: str) -> str:
    return AT_MENTION_RE.sub("", text).strip()


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
    seen: set[str] = set()
    tags: list[str] = []
    for tag in HASHTAG_RE.findall(text):
        tag = tag.lower().strip()
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def count_at_mentions(text: str) -> int:
    return len(AT_MENTION_RE.findall(text))


def contains_spam(text: str) -> bool:
    lower = text.lower()
    return any(spam in lower for spam in SPAM_WORDS)


def contains_promo(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in PROMO_PATTERNS)


def too_generic(text: str) -> bool:
    lower = text.lower()
    return sum(phrase in lower for phrase in GENERIC_PHRASES) >= 2


def excessive_cta(text: str) -> bool:
    lower = text.lower()
    return sum(word in lower for word in CTA_WORDS) >= 2


def repetitive_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 1.0
    return len(set(words)) / len(words)


def emoji_ratio(text: str) -> float:
    if not text:
        return 0.0
    emojis = EMOJI_RE.findall(text)
    return len("".join(emojis)) / len(text)


def is_english_like(text: str) -> bool:
    """
    Two-stage English check.

    Stage 1 — ASCII ratio:
        Rejects scripts like Arabic, Chinese, Hindi, Korean (>10% non-ASCII).

    Stage 2 — English function word frequency:
        Rejects Latin-script non-English text (Spanish, French, etc.)
        that passes Stage 1. A genuine English post has >=10% function words.
    """
    stripped = EMOJI_RE.sub("", text)
    stripped = HASHTAG_RE.sub("", stripped)
    stripped = AT_MENTION_RE.sub("", stripped)
    if not stripped:
        return False

    english_chars = sum(c.isascii() for c in stripped)
    if english_chars / len(stripped) < 0.90:
        return False

    words = re.findall(r"\b[a-z]+\b", stripped.lower())
    if not words:
        return False

    hits = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return (hits / len(words)) >= 0.10


def is_bilingual(text: str) -> bool:
    """
    Detects mixed-language posts (e.g. Norwegian + English, Spanish + English).
    Flags any text where 2+ unambiguous non-English indicator words appear.
    These words were chosen specifically to avoid overlap with English vocabulary.
    """
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    hits = words & NON_ENGLISH_INDICATORS
    return len(hits) >= 2


def topic_relevant(text: str) -> bool:
    lower = text.lower()
    return sum(1 for topic in ALLOWED_TOPICS if topic in lower) >= 2


def ai_pattern_detected(text: str) -> bool:
    lower = text.lower()
    return sum(pattern in lower for pattern in AI_PATTERNS) >= 2


def excessive_at_mentions(text: str) -> bool:
    return count_at_mentions(text) > MAX_AT_MENTIONS


def camelcase_spam(text: str) -> bool:
    """
    Detects inline shadow-hashtag dumps like:
    'BodyTransformation StrengthTraining LegDay MensPhysique WorkoutMotivation'
    These are hashtags written into the caption body without the # symbol.
    """
    matches = CAMELCASE_RE.findall(text)
    return len(matches) > MAX_CAMELCASE


def title_case_abuse(text: str) -> bool:
    """
    Detects posts where Every Single Word Is Capitalized Like This.
    Applies only to texts with enough words to be meaningful.
    """
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    if len(words) < MIN_TITLE_WORDS:
        return False
    title_words = sum(
        1 for w in words
        if w[0].isupper() and w[1:].islower()
    )
    return (title_words / len(words)) > TITLE_CASE_RATIO


# =========================================================
# CLEAN CAPTION
# =========================================================

def clean_caption(caption: str) -> str | None:

    caption = normalize_text(caption)

    # --- Pre-clean checks (run on raw text before stripping) ---

    if excessive_at_mentions(caption):
        return None

    if len(extract_hashtags(caption)) > 8:
        return None

    if camelcase_spam(caption):
        return None

    if title_case_abuse(caption):
        return None

    if is_bilingual(caption):
        return None

    # --- Strip noise ---

    caption = remove_urls(caption)
    caption = remove_at_mentions(caption)
    caption = remove_extra_emojis(caption)
    caption = remove_hashtags(caption)

    # --- Post-clean content checks ---

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
        p for p in urlparse(input_url).path.split("/") if p
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
    keywords: list[str] = []
    seen: set[str] = set()

    hashtags = record.get("hashtags") or []
    if isinstance(hashtags, str):
        hashtags = HASHTAG_RE.findall(hashtags)

    if isinstance(hashtags, list):
        for tag in hashtags:
            tag = str(tag).lower().replace("#", "").strip()
            if not tag or tag in seen or len(tag) < 3:
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

def build_output(caption: str, hashtags: list[str]) -> str:
    hashtags = hashtags[:MAX_HASHTAGS]
    formatted_tags = " ".join(f"#{x}" for x in hashtags)
    return f"{caption}\n\n{formatted_tags}"


def make_record(caption: str, keywords: list[str]) -> dict[str, str]:
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

    clean_records: list[dict[str, str]] = []
    skipped = 0
    seen_outputs: list[str] = []

    rejection_counts: dict[str, int] = {
        "no_caption":        0,
        "at_mentions":       0,
        "too_many_hashtags": 0,
        "camelcase_spam":    0,
        "title_case_abuse":  0,
        "bilingual":         0,
        "spam":              0,
        "promo":             0,
        "excessive_cta":     0,
        "too_generic":       0,
        "ai_pattern":        0,
        "not_english":       0,
        "off_topic":         0,
        "repetitive":        0,
        "emoji_heavy":       0,
        "too_few_sentences": 0,
        "word_count":        0,
        "duplicate":         0,
    }

    for record in tqdm(records):

        raw_caption = record.get("caption")

        if not isinstance(raw_caption, str):
            rejection_counts["no_caption"] += 1
            skipped += 1
            continue

        caption = normalize_text(raw_caption)

        if excessive_at_mentions(caption):
            rejection_counts["at_mentions"] += 1
            skipped += 1
            continue

        if len(extract_hashtags(caption)) > 8:
            rejection_counts["too_many_hashtags"] += 1
            skipped += 1
            continue

        if camelcase_spam(caption):
            rejection_counts["camelcase_spam"] += 1
            skipped += 1
            continue

        if title_case_abuse(caption):
            rejection_counts["title_case_abuse"] += 1
            skipped += 1
            continue

        if is_bilingual(caption):
            rejection_counts["bilingual"] += 1
            skipped += 1
            continue

        caption = remove_urls(caption)
        caption = remove_at_mentions(caption)
        caption = remove_extra_emojis(caption)
        caption = remove_hashtags(caption)

        if contains_spam(caption):
            rejection_counts["spam"] += 1
            skipped += 1
            continue

        if contains_promo(caption):
            rejection_counts["promo"] += 1
            skipped += 1
            continue

        if excessive_cta(caption):
            rejection_counts["excessive_cta"] += 1
            skipped += 1
            continue

        if too_generic(caption):
            rejection_counts["too_generic"] += 1
            skipped += 1
            continue

        if ai_pattern_detected(caption):
            rejection_counts["ai_pattern"] += 1
            skipped += 1
            continue

        if not is_english_like(caption):
            rejection_counts["not_english"] += 1
            skipped += 1
            continue

        if not topic_relevant(caption):
            rejection_counts["off_topic"] += 1
            skipped += 1
            continue

        if repetitive_ratio(caption) < 0.58:
            rejection_counts["repetitive"] += 1
            skipped += 1
            continue

        if emoji_ratio(caption) > 0.12:
            rejection_counts["emoji_heavy"] += 1
            skipped += 1
            continue

        if sentence_count(caption) < MIN_SENTENCES:
            rejection_counts["too_few_sentences"] += 1
            skipped += 1
            continue

        words = word_count(caption)
        if words < MIN_WORDS or words > MAX_WORDS:
            rejection_counts["word_count"] += 1
            skipped += 1
            continue

        is_duplicate = any(
            ratio(caption, existing) > SIMILARITY_THRESHOLD
            for existing in seen_outputs
        )
        if is_duplicate:
            rejection_counts["duplicate"] += 1
            skipped += 1
            continue

        seen_outputs.append(caption)

        niche    = extract_niche(str(record.get("inputUrl", "")))
        keywords = extract_keywords(record, niche)

        if len(keywords) < 3:
            keywords.extend(["fitness", "motivation", "mindset"])

        keywords = keywords[:MAX_HASHTAGS]

        clean_records.append(make_record(caption, keywords))

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)

    print("\n=================================")
    print(f"Records written : {len(clean_records)}")
    print(f"Skipped         : {skipped}")
    print(f"Saved to        : {args.output}")
    print("\n--- Rejection Breakdown ---")
    for reason, count in sorted(
        rejection_counts.items(), key=lambda x: -x[1]
    ):
        if count > 0:
            print(f"  {reason:<22}: {count}")
    print("=================================")


if __name__ == "__main__":
    main()
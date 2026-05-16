"""combine_instagram.py — Consolidate raw Instagram JSON files into a single combined dataset.

Usage:
    python combine_instagram.py [--base-dir insta-dataset] [--output insta-dataset/combined_dataset.json]
"""

import argparse
import json
import sys
from pathlib import Path


def find_source_files(base_dir: Path) -> list[Path]:
    """Return a sorted list of paths matching data-insta-*.json in base_dir."""
    return sorted(base_dir.glob("data-insta-*.json"))


def load_file(path: Path) -> list[dict]:
    """Load a single JSON file and return its records as a list of dicts.

    Returns an empty list and logs an error to stderr if the file contains
    malformed JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(
                f"ERROR: {path}: expected a JSON array, got {type(data).__name__}",
                file=sys.stderr,
            )
            return []
        return data
    except json.JSONDecodeError as exc:
        print(f"ERROR: {path}: malformed JSON — {exc}", file=sys.stderr)
        return []


def deduplicate(records: list[dict]) -> list[dict]:
    """Return records with duplicates removed, keeping the first occurrence by 'id'.

    Records that do not have an 'id' field are always kept (they cannot be
    deduplicated).
    """
    seen_ids: set = set()
    result: list[dict] = []
    for record in records:
        if "id" not in record:
            result.append(record)
            continue
        record_id = record["id"]
        if record_id not in seen_ids:
            seen_ids.add(record_id)
            result.append(record)
    return result


def main() -> None:
    """Entry point for the combine_instagram script."""
    parser = argparse.ArgumentParser(
        description="Combine all insta-dataset/data-insta-*.json files into a single JSON array."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("insta-dataset"),
        help="Directory containing the source data-insta-*.json files (default: insta-dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("insta-dataset/combined_dataset.json"),
        help="Output file path (default: insta-dataset/combined_dataset.json)",
    )
    args = parser.parse_args()

    # Discover source files
    source_files = find_source_files(args.base_dir)
    if not source_files:
        print(
            f"ERROR: No files matching 'data-insta-*.json' found in '{args.base_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load all records from every source file
    all_records: list[dict] = []
    for path in source_files:
        records = load_file(path)
        all_records.extend(records)

    records_loaded = len(all_records)

    # Deduplicate by 'id'
    deduplicated = deduplicate(all_records)
    duplicates_removed = records_loaded - len(deduplicated)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)

    records_written = len(deduplicated)

    # Print summary
    print(f"Files found:        {len(source_files)}")
    print(f"Records loaded:     {records_loaded}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Records written:    {records_written}")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    main()

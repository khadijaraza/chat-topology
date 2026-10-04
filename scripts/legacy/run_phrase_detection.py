#!/usr/bin/env python3
"""Detect corpus-level compound keywords (phrases) before extraction.

Run this between run_ingest.py and run_extraction.py. Its output,
data/processed/legacy/phrases.json, is read by src/extract_keywords.py to merge
detected compounds (e.g. "machine learning") into single keywords instead
of double-counting them as their separate component words -- see
segment_with_phrases in src/extract_keywords.py for how it's used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy.phrase_detection import detect_phrases, get_phrase_config  # noqa: E402

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "phrases.json"

TOP_PREVIEW_COUNT = 30


def main() -> int:
    if not CONVERSATIONS_PATH.is_file():
        print(f"ERROR: Missing input file: {CONVERSATIONS_PATH}", file=sys.stderr)
        print("Run scripts/run_ingest.py first.", file=sys.stderr)
        return 1

    with CONVERSATIONS_PATH.open(encoding="utf-8") as handle:
        conversations = json.load(handle)

    cfg = get_phrase_config()
    phrases = detect_phrases(conversations, config=cfg)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(phrases, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    by_length: dict[int, int] = {}
    for hit in phrases:
        by_length[len(hit["words"])] = by_length.get(len(hit["words"]), 0) + 1

    print("Phrase detection summary")
    print("=" * 40)
    print(f"Conversations scanned: {len(conversations)}")
    print(f"Thresholds: min_pair_count={cfg['min_pair_count']}  min_npmi={cfg['min_npmi']}  "
          f"max_phrase_words={cfg['max_phrase_words']}")
    print(f"Total phrases detected: {len(phrases)}")
    for length in sorted(by_length):
        print(f"  {length}-word phrases: {by_length[length]}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Top {min(TOP_PREVIEW_COUNT, len(phrases))} phrases by count (review before extraction):")
    print("-" * 40)
    for rank, hit in enumerate(phrases[:TOP_PREVIEW_COUNT], start=1):
        print(f"{rank:2d}. {hit['phrase']:<35s}  count={hit['count']:4d}  npmi={hit['npmi']:.3f}")

    if not phrases:
        print()
        print("No phrases cleared the thresholds. If you expected compounds like")
        print("\"machine learning\", try lowering min_pair_count or min_npmi in")
        print("config/legacy/phrase_detection.yaml.")

    print()
    print("BEFORE YOU MOVE ON: skim the list above (or the full JSON). These")
    print("compounds will be merged into single keywords during extraction --")
    print("bad merges here propagate into every downstream node/edge. If")
    print("something looks wrong, tune config/legacy/phrase_detection.yaml and rerun.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

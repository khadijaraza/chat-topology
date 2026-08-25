#!/usr/bin/env python3
"""Extract condensed keywords for all ingested conversations."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_keywords import extract_keywords  # noqa: E402

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chat_keywords.json"


def main() -> int:
    if not CONVERSATIONS_PATH.is_file():
        print(f"ERROR: Missing input file: {CONVERSATIONS_PATH}", file=sys.stderr)
        print("Run scripts/run_ingest.py first.", file=sys.stderr)
        return 1

    with CONVERSATIONS_PATH.open(encoding="utf-8") as handle:
        conversations = json.load(handle)

    results: list[dict[str, object]] = []
    keyword_chat_counts: Counter[str] = Counter()
    total_keyword_occurrences = 0

    for conversation in conversations:
        chat_id = conversation.get("chat_id")
        if not isinstance(chat_id, str):
            continue
        keywords = extract_keywords(conversation)
        results.append({"chat_id": chat_id, "keywords": keywords})
        total_keyword_occurrences += sum(hit["count"] for hit in keywords)
        for hit in keywords:
            keyword_chat_counts[hit["term"]] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    unique_keywords = len(keyword_chat_counts)
    print("Keyword extraction summary")
    print("=" * 40)
    print(f"Conversations processed: {len(results)}")
    print(f"Unique keywords (dataset): {unique_keywords}")
    print(f"Total keyword occurrences: {total_keyword_occurrences}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print("Top 30 keywords by chats appeared in:")
    print("-" * 40)
    for rank, (keyword, chat_count) in enumerate(keyword_chat_counts.most_common(30), start=1):
        print(f"{rank:2d}. {keyword:<35s}  {chat_count:4d} chats")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

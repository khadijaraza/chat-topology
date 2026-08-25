#!/usr/bin/env python3
"""CLI wrapper to ingest raw exports into normalized conversations."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import get_last_ingest_stats, load_all_conversations  # noqa: E402


def conversation_date_range(conversations: list[dict]) -> tuple[str | None, str | None]:
    timestamps = [conv["timestamp"] for conv in conversations if conv.get("timestamp")]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    conversations = load_all_conversations()
    stats = get_last_ingest_stats()

    print("Ingest summary")
    print("=" * 40)

    platforms = sorted(set(stats.loaded_by_platform) | set(stats.skipped_by_platform))
    if not platforms:
        print("No supported platform exports found under data/raw/")
        return 0

    for platform in platforms:
        loaded = stats.loaded_by_platform.get(platform, 0)
        skipped = stats.skipped_by_platform.get(platform, 0)
        print(f"{platform:10s}  loaded: {loaded:5d}  skipped: {skipped:5d}")

    print("-" * 40)
    print(f"{'total':10s}  loaded: {stats.total_loaded:5d}  skipped: {stats.total_skipped:5d}")

    earliest, latest = conversation_date_range(conversations)
    if earliest and latest:
        print(f"\nDate range: {earliest}  →  {latest}")
    else:
        print("\nDate range: (none)")

    print(f"\nOutput: data/processed/conversations.json")

    if stats.warnings:
        print(f"\nWarnings: {len(stats.warnings)} (showing up to 10)")
        for warning in stats.warnings[:10]:
            print(f"  - {warning}")
        if len(stats.warnings) > 10:
            print(f"  ... and {len(stats.warnings) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

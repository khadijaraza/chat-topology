#!/usr/bin/env python3
"""Segment each chat into discourse-cohesive units before extraction.

Run this after run_ingest.py, before run_grounded_extraction.py. Output,
data/processed/segments.json, maps each chat_id to its list of segments
(each with a turn range and a "core"/"tangent" label) -- review before
anything downstream starts depending on segment boundaries being real
topic changes and not noise.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation import get_segmentation_config, segment_conversations  # noqa: E402

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "segments.json"

TANGENT_SAMPLE_COUNT = 15


def main() -> int:
    if not CONVERSATIONS_PATH.is_file():
        print(f"ERROR: Missing input file: {CONVERSATIONS_PATH}", file=sys.stderr)
        print("Run scripts/run_ingest.py first.", file=sys.stderr)
        return 1

    with CONVERSATIONS_PATH.open(encoding="utf-8") as handle:
        conversations = json.load(handle)

    cfg = get_segmentation_config()
    print(
        f"Segmenting {len(conversations)} chats (window_turns={cfg['window_turns']}, "
        f"min_turns_for_segmentation={cfg['min_turns_for_segmentation']})..."
    )
    segments_by_chat = segment_conversations(conversations)

    SEGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEGMENTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(segments_by_chat, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    segment_counts = [len(segs) for segs in segments_by_chat.values()]
    total_chats = len(segments_by_chat) or 1
    multi_segment_chats = sum(1 for c in segment_counts if c > 1)
    distribution = Counter(segment_counts)

    print()
    print("Segmentation summary")
    print("=" * 40)
    print(f"Chats: {len(segments_by_chat)}")
    print(f"Multi-segment chats: {multi_segment_chats} ({multi_segment_chats / total_chats:.0%})")
    print(f"Single-segment (fallback or no shift detected): {len(segments_by_chat) - multi_segment_chats}")
    print(f"Total segments: {sum(segment_counts)}")
    print(
        f"Thresholds: hard_shift_z={cfg['hard_shift_z']}  drift_tolerance={cfg['drift_tolerance']}  "
        f"drift_min_consecutive={cfg['drift_min_consecutive']}"
    )
    print()
    print("Segments-per-chat distribution:")
    print("-" * 40)
    for count in sorted(distribution):
        print(f"  {count:3d} segment(s): {distribution[count]:4d} chats")

    tangent_chats = [
        (chat_id, segs)
        for chat_id, segs in segments_by_chat.items()
        if any(seg["label"] == "tangent" for seg in segs)
    ]
    print()
    print(
        f"Sample of {min(TANGENT_SAMPLE_COUNT, len(tangent_chats))} chats with a tangent detected "
        f"(review before continuing):"
    )
    print("-" * 40)
    for chat_id, segs in tangent_chats[:TANGENT_SAMPLE_COUNT]:
        print(f"  {chat_id}:")
        for seg in segs:
            boundary = f" ({seg['boundary_type']})" if seg["boundary_type"] else ""
            print(
                f"    [{seg['label']:7s}] turns {seg['start_turn']:3d}-{seg['end_turn']:3d}  "
                f"words={seg['word_count']:5d}{boundary}"
            )

    print()
    print("BEFORE YOU MOVE ON: skim the tangent samples above (or the full")
    print("data/processed/segments.json). These boundaries will define what counts")
    print('as "co-occurring" once extraction/graph-building move to segment-level --')
    print("bad boundaries here propagate downstream. Tune thresholds in")
    print("config/segmentation.yaml and rerun if shifts look spurious or real")
    print("tangents aren't being caught.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

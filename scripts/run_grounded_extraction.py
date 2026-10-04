#!/usr/bin/env python3
"""Extract grounded, similarity-ranked keywords for every segment.

Run this after run_segmentation.py. Output, data/processed/segment_keywords.json,
has one entry per segment: {"segment_id", "chat_id", "label", "keywords": [...]}.
Unlike chat_keywords.json's frequency counts, "keywords" here are ranked by
similarity to the segment (grounded, verbatim, MMR-diversified) -- see
src/grounded_extraction.py.

Also writes data/processed/segment_embeddings.npz (segment_ids + their
embedding vectors, computed once here and reused rather than re-embedded) --
the input the proximity-layout pipeline clusters/projects.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grounded_extraction import extract_segment_keywords, get_grounded_extraction_config  # noqa: E402

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "segments.json"
SEGMENT_KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_keywords.json"
SEGMENT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_embeddings.npz"

SAMPLE_COUNT = 20
SNIPPET_CHARS = 220


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (CONVERSATIONS_PATH, "scripts/run_ingest.py"),
            (SEGMENTS_PATH, "scripts/run_segmentation.py"),
        ]
        if not path.is_file()
    ]
    if missing:
        for path, script in missing:
            print(f"ERROR: Missing input file: {path}", file=sys.stderr)
            print(f"Run {script} first.", file=sys.stderr)
        return 1

    with CONVERSATIONS_PATH.open(encoding="utf-8") as handle:
        conversations = json.load(handle)
    with SEGMENTS_PATH.open(encoding="utf-8") as handle:
        segments_by_chat = json.load(handle)

    cfg = get_grounded_extraction_config()
    total_segments = sum(len(segs) for segs in segments_by_chat.values())
    print(f"Extracting grounded keywords for {total_segments} segments across "
          f"{len(segments_by_chat)} chats (top_k={cfg['top_k_per_segment']}, "
          f"mmr_lambda={cfg['mmr_lambda']})...")

    results, segment_embeddings = extract_segment_keywords(conversations, segments_by_chat)

    SEGMENT_KEYWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEGMENT_KEYWORDS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    segment_ids = np.array(list(segment_embeddings.keys()))
    embeddings = np.stack(list(segment_embeddings.values())).astype(np.float32) if segment_embeddings else np.empty((0, 0), dtype=np.float32)
    np.savez(SEGMENT_EMBEDDINGS_PATH, segment_ids=segment_ids, embeddings=embeddings)

    unique_keywords: set[str] = set()
    empty_segments = 0
    keyword_counts_per_segment = []
    for entry in results:
        terms = [hit["term"] for hit in entry["keywords"]]
        unique_keywords.update(terms)
        keyword_counts_per_segment.append(len(terms))
        if not terms:
            empty_segments += 1

    print()
    print("Grounded extraction summary")
    print("=" * 40)
    print(f"Segments processed: {len(results)}")
    print(f"Segments with zero candidates: {empty_segments}")
    print(f"Segment embeddings persisted: {len(segment_embeddings)}")
    print(f"Unique keywords across corpus: {len(unique_keywords)}")
    print(f"Output: {SEGMENT_KEYWORDS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Embeddings: {SEGMENT_EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)}")

    # Build a lookup of segment text so the sample can show real words
    # alongside the extracted keywords -- can't confirm quality from
    # counts/labels alone.
    turns_by_chat = {conv["chat_id"]: conv.get("turns", []) for conv in conversations}
    segment_by_id = {seg["segment_id"]: seg for segs in segments_by_chat.values() for seg in segs}

    non_empty = [entry for entry in results if entry["keywords"]]
    random.seed(42)
    sample = random.sample(non_empty, min(SAMPLE_COUNT, len(non_empty)))

    print()
    print(f"Sample of {len(sample)} segments -- keywords next to the actual text "
          f"(review before continuing):")
    print("-" * 40)
    for entry in sample:
        segment = segment_by_id.get(entry["segment_id"])
        if segment is None:
            continue
        turns = turns_by_chat.get(entry["chat_id"], [])
        text = "\n".join(
            t["text"] for t in turns[segment["start_turn"]:segment["end_turn"]] if isinstance(t.get("text"), str)
        )
        snippet = text[:SNIPPET_CHARS].replace("\n", " ")
        if len(text) > SNIPPET_CHARS:
            snippet += "..."
        terms = ", ".join(
            f"{hit['term']}{'[' + hit['entity_label'] + ']' if hit.get('entity_label') else ''} "
            f"({hit['score']:.2f})"
            for hit in entry["keywords"]
        )
        print(f"  {entry['segment_id']}  [{entry['label']}]")
        print(f"    text: {snippet}")
        print(f"    keywords: {terms}")
        print()

    print("BEFORE YOU MOVE ON: read the sample above -- do the extracted keywords")
    print("actually summarize what that segment is about? Grounded/MMR ranking can")
    print("still surface a technically-verbatim but unrepresentative phrase if the")
    print("candidate pool is thin. Tune top_k_per_segment/mmr_lambda in")
    print("config/grounded_extraction.yaml and rerun if quality looks off. Nothing")
    print("downstream (graph build, height scores) has been rewired to use this yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

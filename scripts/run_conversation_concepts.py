#!/usr/bin/env python3
"""Group segments into conversation-level concepts (mid tier of the
proximity visualization's 3-layer hierarchy).

Run this after run_grounded_extraction.py. Output,
data/processed/conversation_concepts.json, maps each concept label to its
member segment ids -- review the real text before anything downstream
(macro domains, the 2D projection) depends on these groupings.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation_concepts import (  # noqa: E402
    cluster_segments_into_concepts,
    get_conversation_concepts_config,
    load_segment_embeddings,
)

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "segments.json"
SEGMENT_KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_keywords.json"
SEGMENT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_embeddings.npz"
CONVERSATION_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "conversation_concepts.json"

TOP_PREVIEW_COUNT = 25
MEMBER_SAMPLE_COUNT = 3
SNIPPET_CHARS = 160


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (CONVERSATIONS_PATH, "scripts/run_ingest.py"),
            (SEGMENTS_PATH, "scripts/run_segmentation.py"),
            (SEGMENT_KEYWORDS_PATH, "scripts/run_grounded_extraction.py"),
            (SEGMENT_EMBEDDINGS_PATH, "scripts/run_grounded_extraction.py"),
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
    with SEGMENT_KEYWORDS_PATH.open(encoding="utf-8") as handle:
        segment_keywords = json.load(handle)

    segment_ids, embeddings = load_segment_embeddings(SEGMENT_EMBEDDINGS_PATH)
    top_keyword_by_segment = {
        entry["segment_id"]: entry["keywords"][0]["term"]
        for entry in segment_keywords
        if entry.get("keywords")
    }

    cfg = get_conversation_concepts_config()
    print(f"Clustering {len(segment_ids)} segments (min_cluster_size={cfg['min_cluster_size']}, "
          f"min_samples={cfg['min_samples']})...")
    segment_to_concept, concept_to_segments = cluster_segments_into_concepts(
        segment_ids, embeddings, top_keyword_by_segment
    )

    CONVERSATION_CONCEPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONVERSATION_CONCEPTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {label: {"segment_ids": sorted(members)} for label, members in concept_to_segments.items()},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    total_segments = len(segment_ids)
    singleton_concepts = sum(1 for members in concept_to_segments.values() if len(members) == 1)
    grouped_concepts = len(concept_to_segments) - singleton_concepts
    grouped_segments = total_segments - singleton_concepts

    print()
    print("Conversation concepts summary")
    print("=" * 40)
    print(f"Segments in: {total_segments}")
    print(f"Concepts out: {len(concept_to_segments)}  "
          f"({grouped_concepts} real clusters, {singleton_concepts} singleton/unclustered)")
    if total_segments:
        print(f"Segments merged into a real cluster: {grouped_segments} ({grouped_segments / total_segments:.0%})")
    print(f"Output: {CONVERSATION_CONCEPTS_PATH.relative_to(PROJECT_ROOT)}")

    # Real text next to cluster assignments -- counts/labels alone can't
    # confirm whether segments were grouped sensibly.
    turns_by_chat = {conv["chat_id"]: conv.get("turns", []) for conv in conversations}
    segment_by_id = {seg["segment_id"]: seg for segs in segments_by_chat.values() for seg in segs}

    def snippet_for(segment_id: str) -> str:
        segment = segment_by_id.get(segment_id)
        if segment is None:
            return "(segment not found)"
        turns = turns_by_chat.get(segment["chat_id"], [])
        text = "\n".join(
            t["text"] for t in turns[segment["start_turn"]:segment["end_turn"]] if isinstance(t.get("text"), str)
        )
        text = text[:SNIPPET_CHARS].replace("\n", " ")
        return text + "..." if len(text) >= SNIPPET_CHARS else text

    ranked = sorted(
        ((label, members) for label, members in concept_to_segments.items() if len(members) > 1),
        key=lambda item: -len(item[1]),
    )
    random.seed(11)
    print()
    print(f"Top {min(TOP_PREVIEW_COUNT, len(ranked))} real concepts by member count "
          f"(review before continuing):")
    print("-" * 40)
    for rank, (label, members) in enumerate(ranked[:TOP_PREVIEW_COUNT], start=1):
        print(f"{rank:2d}. {label:<30s} members={len(members):4d}")
        sample = random.sample(members, min(MEMBER_SAMPLE_COUNT, len(members)))
        for segment_id in sample:
            print(f"      [{segment_id}] {snippet_for(segment_id)}")

    print()
    print("BEFORE YOU MOVE ON: do the sampled segment texts under each concept actually")
    print("read as the same topic neighborhood? Tune min_cluster_size/min_samples in")
    print("config/conversation_concepts.yaml and rerun if groupings look wrong. Nothing")
    print("downstream (macro domains, the 2D projection) has been built on this yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

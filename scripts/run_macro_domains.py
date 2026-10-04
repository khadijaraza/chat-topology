#!/usr/bin/env python3
"""Group segments into macro domains via mutual-kNN + Leiden (CPM) community
detection + TF-ICF labeling (top tier of the proximity visualization's
3-layer hierarchy).

Run this after run_grounded_extraction.py and run_conversation_concepts.py.
Clusters segments directly (not conversation-concept centroids) -- see
src/macro_domains.py's module docstring for why. Output,
data/processed/macro_domains.json, maps each domain label to its member
segment ids (+ which conversation concepts land there by majority vote) --
review the real text before the 2D projection is built on top of this.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation_concepts import load_segment_embeddings  # noqa: E402
from src.macro_domains import (  # noqa: E402
    BACKGROUND_LABEL,
    assign_concepts_to_domains,
    cluster_into_macro_domains,
    get_macro_domains_config,
)

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "segments.json"
SEGMENT_KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_keywords.json"
SEGMENT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_embeddings.npz"
CONVERSATION_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "conversation_concepts.json"
MACRO_DOMAINS_PATH = PROJECT_ROOT / "data" / "processed" / "macro_domains.json"

SEGMENT_SAMPLE_PER_DOMAIN = 4
SNIPPET_CHARS = 140
DOMAIN_PREVIEW_COUNT = 40


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (CONVERSATIONS_PATH, "scripts/run_ingest.py"),
            (SEGMENTS_PATH, "scripts/run_segmentation.py"),
            (SEGMENT_KEYWORDS_PATH, "scripts/run_grounded_extraction.py"),
            (SEGMENT_EMBEDDINGS_PATH, "scripts/run_grounded_extraction.py"),
            (CONVERSATION_CONCEPTS_PATH, "scripts/run_conversation_concepts.py"),
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
        segment_keywords_raw = json.load(handle)
    with CONVERSATION_CONCEPTS_PATH.open(encoding="utf-8") as handle:
        concept_summary = json.load(handle)

    segment_ids, embeddings = load_segment_embeddings(SEGMENT_EMBEDDINGS_PATH)
    segment_keywords = {
        entry["segment_id"]: [kw["term"] for kw in entry.get("keywords", [])]
        for entry in segment_keywords_raw
    }
    concept_to_segments = {label: info["segment_ids"] for label, info in concept_summary.items()}

    cfg = get_macro_domains_config()
    print(
        f"Clustering {len(segment_ids)} segments via mutual-kNN (k={cfg['knn_k']}) + "
        f"Leiden/CPM (resolution={cfg['cpm_resolution']})..."
    )
    segment_to_domain, domain_to_segments = cluster_into_macro_domains(
        segment_ids, embeddings, segment_keywords
    )
    concept_to_domain = assign_concepts_to_domains(concept_to_segments, segment_to_domain)

    concepts_by_domain: dict[str, list[str]] = {}
    for concept_label, domain_label in concept_to_domain.items():
        concepts_by_domain.setdefault(domain_label, []).append(concept_label)

    MACRO_DOMAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MACRO_DOMAINS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                label: {
                    "segment_ids": sorted(members),
                    "concepts": sorted(concepts_by_domain.get(label, [])),
                    "segment_count": len(members),
                }
                for label, members in domain_to_segments.items()
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    real_domains = {label: members for label, members in domain_to_segments.items() if label != BACKGROUND_LABEL}
    background_count = len(domain_to_segments.get(BACKGROUND_LABEL, []))

    print()
    print("Macro domains summary")
    print("=" * 40)
    print(f"Segments in: {len(segment_ids)}")
    print(f"Real domains: {len(real_domains)}")
    print(f"Background / Minor Orbit: {background_count} segments "
          f"({background_count / len(segment_ids):.0%} of corpus)")
    print(f"Output: {MACRO_DOMAINS_PATH.relative_to(PROJECT_ROOT)}")

    sizes = sorted((len(members) for members in real_domains.values()), reverse=True)
    if sizes:
        print(f"Domain size distribution: min={sizes[-1]} max={sizes[0]} "
              f"median={sizes[len(sizes) // 2]}")

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

    ranked = sorted(real_domains.items(), key=lambda item: -len(item[1]))
    random.seed(23)
    print()
    print(f"Top {min(DOMAIN_PREVIEW_COUNT, len(ranked))} domains, ranked by segment count "
          f"(review before continuing):")
    print("-" * 40)
    for rank, (label, members) in enumerate(ranked[:DOMAIN_PREVIEW_COUNT], start=1):
        print(f"{rank:2d}. {label:<60s} segments={len(members):4d}")
        sample = random.sample(members, min(SEGMENT_SAMPLE_PER_DOMAIN, len(members)))
        for segment_id in sample:
            print(f"      [{segment_id}] {snippet_for(segment_id)}")

    if background_count:
        print()
        print(f"Background / Minor Orbit sample (of {background_count}):")
        sample = random.sample(domain_to_segments[BACKGROUND_LABEL], min(SEGMENT_SAMPLE_PER_DOMAIN, background_count))
        for segment_id in sample:
            print(f"      [{segment_id}] {snippet_for(segment_id)}")

    print()
    print("BEFORE YOU MOVE ON: do the sampled segments under each domain share a real")
    print("conceptual territory -- not just a broad topic, an actually coherent one? Are")
    print("labels (top TF-ICF terms) representative of their domain's content? Is the")
    print("Background bucket plausible (one-off tangents, not a big chunk of real signal")
    print("being discarded)? Tune knn_k/cpm_resolution/min_community_size/")
    print("min_avg_similarity in config/macro_domains.yaml and rerun if domain count is")
    print("far outside ~25-45, a domain reads as incoherent, or Background looks wrong.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

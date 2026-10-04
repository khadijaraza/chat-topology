#!/usr/bin/env python3
"""Cluster segment-level grounded keywords (Phase 2 output) into concepts.

Run this after run_grounded_extraction.py. Phase 3 of the pipeline
reconstruction -- see CLAUDE.md. Itself ALSO now superseded by
src/conversation_concepts.py (the current pipeline's mid tier, which
clusters segment EMBEDDINGS directly rather than segment keywords) -- kept
for the same "superseded, not deleted" reason as the rest of scripts/legacy/,
not part of either pipeline's normal run order. Unlike the original
run_concept_clustering.py (which embeds bare decontextualized terms from
data/processed/legacy/chat_keywords.json), this embeds each term WITH
lightweight co-occurrence context from its best-scoring segment, to close
the polysemy gap that a stronger embedding model alone didn't fully close.

Writes NEW files alongside the originals rather than overwriting them
(data/processed/legacy/concepts.json, chat_concepts.json still exist and are
still what graph_build/supertopic_clustering currently read) -- nothing
downstream has been rewired to use this yet; that's Phase 4.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy.concept_clustering import (  # noqa: E402
    build_segment_concepts,
    cluster_segment_keywords_into_concepts,
)

SEGMENT_KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_keywords.json"
SEGMENT_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "segment_concepts.json"
CONCEPT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "segment_concept_summary.json"

TOP_PREVIEW_COUNT = 25
MEMBERS_PREVIEW_COUNT = 12


def main() -> int:
    if not SEGMENT_KEYWORDS_PATH.is_file():
        print(f"ERROR: Missing input file: {SEGMENT_KEYWORDS_PATH}", file=sys.stderr)
        print("Run scripts/run_grounded_extraction.py first.", file=sys.stderr)
        return 1

    with SEGMENT_KEYWORDS_PATH.open(encoding="utf-8") as handle:
        segment_keywords = json.load(handle)

    print("Embedding (with co-occurrence context) + clustering segment keywords...")
    term_to_concept, concept_to_terms = cluster_segment_keywords_into_concepts(segment_keywords)

    segment_concepts = build_segment_concepts(segment_keywords, term_to_concept)

    SEGMENT_CONCEPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEGMENT_CONCEPTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(segment_concepts, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    concept_chat_ids: dict[str, set[str]] = {}
    for entry in segment_concepts:
        chat_id = entry.get("chat_id")
        for hit in entry.get("keywords", []):
            concept_chat_ids.setdefault(hit["term"], set()).add(chat_id)
    concept_chat_counts = {label: len(ids) for label, ids in concept_chat_ids.items()}

    summary = {
        label: {"chat_count": concept_chat_counts.get(label, 0), "members": sorted(members)}
        for label, members in concept_to_terms.items()
    }
    with CONCEPT_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    total_terms = len(term_to_concept)
    singleton_concepts = sum(1 for members in concept_to_terms.values() if len(members) == 1)
    grouped_concepts = len(concept_to_terms) - singleton_concepts
    grouped_terms = total_terms - singleton_concepts

    print()
    print("Segment concept clustering summary")
    print("=" * 40)
    print(f"Distinct keywords in: {total_terms}")
    print(f"Concepts out: {len(concept_to_terms)}  "
          f"({grouped_concepts} real clusters, {singleton_concepts} singleton/unclustered)")
    print(f"Keywords merged into a real cluster: {grouped_terms} "
          f"({grouped_terms / total_terms:.0%})" if total_terms else "")
    print(f"Segment-concepts output: {SEGMENT_CONCEPTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Concept summary: {CONCEPT_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")

    ranked = sorted(
        ((label, members) for label, members in concept_to_terms.items() if len(members) > 1),
        key=lambda item: -concept_chat_counts.get(item[0], 0),
    )
    print()
    print(f"Top {min(TOP_PREVIEW_COUNT, len(ranked))} real concepts by chat_count "
          f"(review before continuing):")
    print("-" * 40)
    for rank, (label, members) in enumerate(ranked[:TOP_PREVIEW_COUNT], start=1):
        chat_count = concept_chat_counts.get(label, 0)
        shown = sorted(members, key=lambda t: t != label)[:MEMBERS_PREVIEW_COUNT]
        extra = (
            f" (+{len(members) - MEMBERS_PREVIEW_COUNT} more)"
            if len(members) > MEMBERS_PREVIEW_COUNT
            else ""
        )
        print(f"{rank:2d}. {label:<25s} chat_count={chat_count:4d}  members={len(members):3d}")
        print(f"      {', '.join(shown)}{extra}")

    # Spot-check the specific polysemy scenario this phase targets: does a
    # bare single word land near its co-occurring sense, not a homonym?
    random.seed(7)
    single_word_terms = [t for t in term_to_concept if " " not in t]
    sample_terms = random.sample(single_word_terms, min(10, len(single_word_terms)))
    print()
    print(f"Sample of {len(sample_terms)} single-word terms and their concept "
          f"(context-disambiguated -- spot-check for polysemy collisions):")
    print("-" * 40)
    for term in sample_terms:
        concept = term_to_concept[term]
        print(f"  {term:<20s} -> {concept}")

    print()
    print("BEFORE YOU MOVE ON: skim the groupings above (or the full")
    print("data/processed/legacy/segment_concept_summary.json). Compare against")
    print("data/processed/legacy/concepts.json (the original chat-keyed clustering) if you")
    print("want to judge whether context-aware embedding meaningfully changed")
    print("quality. Nothing downstream has been rewired to use this yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

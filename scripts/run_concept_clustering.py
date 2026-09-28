#!/usr/bin/env python3
"""Group extracted keywords into semantic concepts.

Run this between run_extraction.py and run_graph_build.py. Its output,
data/processed/chat_concepts.json, has the same shape as chat_keywords.json
but with each keyword hit re-keyed to its concept label -- graph build then
treats concepts exactly like it used to treat raw keywords, so node
identity (and every downstream stage) now runs on semantic groups instead
of individual keywords.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.concept_clustering import (  # noqa: E402
    build_chat_concepts,
    cluster_keywords_into_concepts,
    get_concept_config,
)

KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "chat_keywords.json"
CHAT_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "chat_concepts.json"
CONCEPTS_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "concepts.json"

TOP_PREVIEW_COUNT = 25
MEMBERS_PREVIEW_COUNT = 12


def main() -> int:
    if not KEYWORDS_PATH.is_file():
        print(f"ERROR: Missing input file: {KEYWORDS_PATH}", file=sys.stderr)
        print("Run scripts/run_extraction.py first.", file=sys.stderr)
        return 1

    with KEYWORDS_PATH.open(encoding="utf-8") as handle:
        chat_keywords = json.load(handle)

    cfg = get_concept_config()
    print("Embedding + clustering keywords (this downloads the model on first run)...")
    term_to_concept, concept_to_terms = cluster_keywords_into_concepts(chat_keywords)

    chat_concepts = build_chat_concepts(chat_keywords, term_to_concept)

    CHAT_CONCEPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_CONCEPTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chat_concepts, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    concept_chat_counts: Counter[str] = Counter()
    for entry in chat_concepts:
        for hit in entry.get("keywords", []):
            concept_chat_counts[hit["term"]] += 1

    summary = {
        label: {"chat_count": concept_chat_counts.get(label, 0), "members": sorted(members)}
        for label, members in concept_to_terms.items()
    }
    with CONCEPTS_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    total_terms = len(term_to_concept)
    singleton_concepts = sum(1 for members in concept_to_terms.values() if len(members) == 1)
    grouped_concepts = len(concept_to_terms) - singleton_concepts
    grouped_terms = total_terms - singleton_concepts

    print()
    print("Concept clustering summary")
    print("=" * 40)
    print(f"Model: {cfg['model_name']}  min_cluster_size={cfg['min_cluster_size']}  "
          f"min_samples={cfg['min_samples']}")
    print(f"Distinct keywords in: {total_terms}")
    print(f"Concepts out: {len(concept_to_terms)}  "
          f"({grouped_concepts} real clusters, {singleton_concepts} singleton/unclustered)")
    print(f"Keywords merged into a real cluster: {grouped_terms} "
          f"({grouped_terms / total_terms:.0%})")
    print(f"Chat-concept output: {CHAT_CONCEPTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Concept summary: {CONCEPTS_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")

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
        extra = f" (+{len(members) - MEMBERS_PREVIEW_COUNT} more)" if len(members) > MEMBERS_PREVIEW_COUNT else ""
        print(f"{rank:2d}. {label:<25s} chat_count={chat_count:4d}  members={len(members):3d}")
        print(f"      {', '.join(shown)}{extra}")

    print()
    print("BEFORE YOU MOVE ON: skim the groupings above (or the full")
    print("data/processed/concepts.json). These clusters become graph nodes --")
    print("keywords that shouldn't be together (or should be but aren't) will")
    print("propagate into every downstream node/edge. Tune min_cluster_size /")
    print("min_samples in config/concept_clustering.yaml and rerun if something")
    print("looks wrong. Nothing downstream has been rewired to use this yet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

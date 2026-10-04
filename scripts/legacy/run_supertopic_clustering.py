#!/usr/bin/env python3
"""Group concept nodes into <100 broader super-topics for graph legibility.

Run this after run_graph_build.py -- deliberately reads graph.gpickle's
node list, NOT concepts.json directly. concepts.json is the full,
unfiltered output of concept clustering (tens of thousands of concepts,
most singletons that occur in only 1-2 chats); graph.gpickle's nodes are
what's left after config/legacy/graph.yaml's min_chat_count filter, i.e. the
actual ~5-6k concepts that become visible graph nodes. Clustering the
unfiltered set pulls in massive amounts of one-off noise the graph never
shows anyway. Output, data/processed/legacy/supertopics.json, maps each
super-topic label to its member concept labels -- review before it's used
to organize/color/drill-down the frontend. Every concept is force-assigned
to exactly one super-topic (see src/supertopic_clustering.py for why
there's no "noise" bucket here).
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy.supertopic_clustering import cluster_into_supertopics, get_supertopic_config  # noqa: E402

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "graph.gpickle"
SUPERTOPICS_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "supertopics.json"

MEMBERS_PREVIEW_COUNT = 15


def main() -> int:
    if not GRAPH_PATH.is_file():
        print(f"ERROR: Missing input file: {GRAPH_PATH}", file=sys.stderr)
        print("Run scripts/legacy/run_graph_build.py first.", file=sys.stderr)
        return 1

    with GRAPH_PATH.open("rb") as handle:
        graph = pickle.load(handle)

    concept_labels = sorted(graph.nodes)
    concept_chat_counts: Counter[str] = Counter(
        {label: graph.nodes[label].get("chat_count", 0) for label in concept_labels}
    )

    cfg = get_supertopic_config()
    print(f"Embedding + clustering {len(concept_labels)} concepts into <= {cfg['n_clusters']} super-topics...")
    concept_to_supertopic, supertopic_to_concepts = cluster_into_supertopics(
        concept_labels, concept_chat_counts
    )

    summary = {
        label: {
            "chat_count": sum(concept_chat_counts.get(c, 0) for c in members),
            "concept_count": len(members),
            "members": sorted(members),
        }
        for label, members in supertopic_to_concepts.items()
    }
    SUPERTOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUPERTOPICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print()
    print("Super-topic clustering summary")
    print("=" * 40)
    print(f"Model: {cfg['model_name']}  n_clusters={cfg['n_clusters']}  linkage={cfg['linkage']}")
    print(f"Concepts in: {len(concept_labels)}")
    print(f"Super-topics out: {len(supertopic_to_concepts)}")
    print(f"Output: {SUPERTOPICS_PATH.relative_to(PROJECT_ROOT)}")

    ranked = sorted(summary.items(), key=lambda item: -item[1]["chat_count"])
    print()
    print(f"All {len(ranked)} super-topics by aggregate chat_count (review before continuing):")
    print("-" * 40)
    for rank, (label, info) in enumerate(ranked, start=1):
        shown = info["members"][:MEMBERS_PREVIEW_COUNT]
        extra = (
            f" (+{info['concept_count'] - MEMBERS_PREVIEW_COUNT} more)"
            if info["concept_count"] > MEMBERS_PREVIEW_COUNT
            else ""
        )
        print(f"{rank:2d}. {label:<25s} chat_count={info['chat_count']:5d}  concepts={info['concept_count']:4d}")
        print(f"      {', '.join(shown)}{extra}")

    print()
    print("BEFORE YOU MOVE ON: skim the groupings above (or the full")
    print("data/processed/legacy/supertopics.json). Every concept is forced into some")
    print("super-topic here (no unclustered/'noise' bucket), so a few may look")
    print("like an arbitrary catch-all -- that's expected with a hard cluster-")
    print("count cap. Tune n_clusters/linkage in config/legacy/supertopic_clustering.yaml")
    print("and rerun if the groupings don't look organized enough to be useful.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

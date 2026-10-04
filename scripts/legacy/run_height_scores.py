#!/usr/bin/env python3
"""Compute per-topic height (Z-axis) scores from graph + conversation data."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.height_score import compute_height_scores, get_height_config  # noqa: E402

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "chat_concepts.json"
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "graph.gpickle"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "height_scores.json"

TOP_PREVIEW_COUNT = 15


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (CONVERSATIONS_PATH, "scripts/run_ingest.py"),
            (CONCEPTS_PATH, "scripts/legacy/run_concept_clustering.py"),
            (GRAPH_PATH, "scripts/legacy/run_graph_build.py"),
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
    with CONCEPTS_PATH.open(encoding="utf-8") as handle:
        chat_concepts = json.load(handle)
    with GRAPH_PATH.open("rb") as handle:
        graph = pickle.load(handle)

    scores = compute_height_scores(graph, conversations, chat_concepts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    weights, z_range = get_height_config()
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    print("Height score summary")
    print("=" * 40)
    print(f"Topics scored: {len(scores)}")
    print(
        f"Weights: frequency={weights['frequency']}  depth={weights['depth']}  "
        f"sustained_focus={weights['sustained_focus']}  (config/height_weights.yaml)"
    )
    print(f"Z range: [0, {z_range}]")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    if ranked:
        print()
        print(f"Top {min(TOP_PREVIEW_COUNT, len(ranked))} topics by height score:")
        print("-" * 40)
        for rank, (term, score) in enumerate(ranked[:TOP_PREVIEW_COUNT], start=1):
            print(f"{rank:2d}. {term:<35s}  z={score:.3f}")

        print()
        bottom = ranked[-TOP_PREVIEW_COUNT:][::-1]
        print(f"Bottom {len(bottom)} topics by height score:")
        print("-" * 40)
        for rank, (term, score) in enumerate(bottom, start=1):
            print(f"{rank:2d}. {term:<35s}  z={score:.3f}")

    print()
    print("BEFORE YOU MOVE ON: check that the top list is dominated by topics")
    print("you'd actually expect to be deep/frequent, not junk keywords with a")
    print("single freakishly long chat. If sustained_focus is overpowering")
    print("everything, tune the weights in config/height_weights.yaml and rerun --")
    print("no code changes needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and save the concept co-occurrence graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_graph import build_cooccurrence_graph, get_graph_config, top_weighted_edges  # noqa: E402

CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "chat_concepts.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "graph.gpickle"


def save_graph(graph, path: Path) -> str:
    """Persist graph to disk. Returns a short format description."""
    import pickle

    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return "gpickle (stdlib pickle; NetworkX 3.x removed nx.write_gpickle)"


def main() -> int:
    config_min_chat_count, config_max_edges_per_node = get_graph_config()
    parser = argparse.ArgumentParser(description="Build keyword co-occurrence graph.")
    parser.add_argument(
        "--min-chat-count",
        type=int,
        default=None,
        help=f"Drop keywords appearing in fewer than this many chats "
        f"(default: config/graph.yaml's min_chat_count={config_min_chat_count}).",
    )
    parser.add_argument(
        "--max-edges-per-node",
        type=int,
        default=None,
        help="Keep only each node's strongest N edges by weight "
        f"(default: config/graph.yaml's max_edges_per_node={config_max_edges_per_node}).",
    )
    args = parser.parse_args()

    if not CONCEPTS_PATH.is_file():
        print(f"ERROR: Missing input file: {CONCEPTS_PATH}", file=sys.stderr)
        print("Run scripts/run_concept_clustering.py first.", file=sys.stderr)
        return 1

    with CONCEPTS_PATH.open(encoding="utf-8") as handle:
        chat_concepts = json.load(handle)

    graph = build_cooccurrence_graph(
        chat_concepts,
        min_chat_count=args.min_chat_count,
        max_edges_per_node=args.max_edges_per_node,
    )
    min_chat_count = args.min_chat_count if args.min_chat_count is not None else config_min_chat_count
    max_edges_per_node = (
        args.max_edges_per_node if args.max_edges_per_node is not None else config_max_edges_per_node
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = save_graph(graph, OUTPUT_PATH)

    print("Graph build summary")
    print("=" * 40)
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Min chat count filter: {min_chat_count}")
    print(f"Max edges per node: {max_edges_per_node}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Format: {fmt}")
    print("  - Preserves float edge attrs (PMI) exactly for this Python pipeline.")
    print("  - Prefer GraphML if you need cross-language interchange.")
    print()
    print("Top 10 edges by weight:")
    print("-" * 40)
    for rank, (term_a, term_b, weight) in enumerate(top_weighted_edges(graph, 10), start=1):
        edge_data = graph.edges[term_a, term_b]
        print(
            f"{rank:2d}. {term_a} ↔ {term_b}  "
            f"weight={weight:.3f}  "
            f"(cooccurrence={edge_data['cooccurrence']}, pmi={edge_data['pmi']:.3f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

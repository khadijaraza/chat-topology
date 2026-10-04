#!/usr/bin/env python3
"""Compute the keyword graph layout and save a static preview PNG."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display backend needed
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout import compute_layout, get_coordinate_range  # noqa: E402

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "graph.gpickle"
LAYOUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "layout.json"
PREVIEW_PNG_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "layout_preview.png"

TOP_LABEL_COUNT = 20


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_layout_json(layout: dict[str, tuple[float, float]], path: Path) -> None:
    payload = {term: [x, y] for term, (x, y) in layout.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def top_terms_by_chat_count(graph, limit: int) -> list[str]:
    ranked = sorted(
        graph.nodes(data=True),
        key=lambda item: (-item[1].get("chat_count", 0), item[0]),
    )
    return [term for term, _ in ranked[:limit]]


def render_preview(
    graph,
    layout: dict[str, tuple[float, float]],
    labeled_terms: set[str],
    coordinate_range: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    xs = [layout[term][0] for term in graph.nodes if term in layout]
    ys = [layout[term][1] for term in graph.nodes if term in layout]

    # All nodes: small, low-emphasis circles so overall cluster shape is
    # visible without trying to read every label.
    ax.scatter(xs, ys, s=15, c="#8ca0c4", alpha=0.5, linewidths=0, zorder=1)

    # Highlight + label only the top-N by chat_count, so the preview stays
    # readable.
    label_xs, label_ys = [], []
    for term in labeled_terms:
        if term not in layout:
            continue
        x, y = layout[term]
        label_xs.append(x)
        label_ys.append(y)
        ax.annotate(
            term,
            (x, y),
            fontsize=8,
            fontweight="bold",
            color="#1a1a2e",
            xytext=(4, 4),
            textcoords="offset points",
            zorder=3,
        )
    ax.scatter(label_xs, label_ys, s=60, c="#e94560", alpha=0.9, linewidths=0, zorder=2)

    margin = coordinate_range * 1.15
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.set_title(
        f"Keyword co-occurrence layout preview "
        f"(top {len(labeled_terms)} by chat_count labeled, of {graph.number_of_nodes()} nodes)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    if not GRAPH_PATH.is_file():
        print(f"ERROR: Missing input file: {GRAPH_PATH}", file=sys.stderr)
        print("Run scripts/legacy/run_graph_build.py first.", file=sys.stderr)
        return 1

    graph = load_graph(GRAPH_PATH)

    if graph.number_of_nodes() == 0:
        print("ERROR: Graph has no nodes; nothing to lay out.", file=sys.stderr)
        return 1

    layout = compute_layout(graph)
    coordinate_range = get_coordinate_range()

    save_layout_json(layout, LAYOUT_JSON_PATH)

    labeled_terms = set(top_terms_by_chat_count(graph, TOP_LABEL_COUNT))
    render_preview(graph, layout, labeled_terms, coordinate_range, PREVIEW_PNG_PATH)

    print("Layout summary")
    print("=" * 40)
    print(f"Nodes laid out: {len(layout)}")
    print(f"Coordinate range: [-{coordinate_range:g}, {coordinate_range:g}] (config/layout.yaml)")
    print(f"Labeled in preview (top {TOP_LABEL_COUNT} by chat_count):")
    for rank, term in enumerate(sorted(labeled_terms, key=lambda t: -graph.nodes[t].get("chat_count", 0)), start=1):
        print(f"  {rank:2d}. {term:<30s} chat_count={graph.nodes[term].get('chat_count', 0)}")
    print()
    print(f"Layout JSON: {LAYOUT_JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Preview PNG: {PREVIEW_PNG_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

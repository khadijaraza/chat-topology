#!/usr/bin/env python3
"""Render an interactive, pannable/zoomable HTML preview of the keyword layout.

Unlike run_layout.py's static PNG (good for a quick headless sanity check),
this opens in a browser: scroll/drag to zoom and pan, and hover any point to
see its term, chat_count, and (optionally) its top co-occurring neighbors.
Useful once the node count is too high for a static image to be readable.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph.gpickle"
LAYOUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "layout.json"
OUTPUT_HTML_PATH = PROJECT_ROOT / "data" / "processed" / "layout_interactive.html"

MIN_MARKER_SIZE = 4.0
MAX_MARKER_SIZE = 22.0


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_layout(path: Path) -> dict[str, tuple[float, float]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {term: (float(xy[0]), float(xy[1])) for term, xy in raw.items()}


def scaled_marker_sizes(chat_counts: list[int]) -> list[float]:
    """Map chat_count -> marker size on a log scale (counts are power-law skewed)."""
    if not chat_counts:
        return []
    log_counts = [math.log1p(c) for c in chat_counts]
    lo, hi = min(log_counts), max(log_counts)
    if hi <= lo:
        return [MIN_MARKER_SIZE for _ in chat_counts]
    span = hi - lo
    return [
        MIN_MARKER_SIZE + (lc - lo) / span * (MAX_MARKER_SIZE - MIN_MARKER_SIZE)
        for lc in log_counts
    ]


def top_neighbors(graph, term: str, limit: int = 5) -> str:
    if term not in graph:
        return ""
    edges = [
        (other, data.get("weight", 0.0))
        for other, data in graph.adj[term].items()
    ]
    edges.sort(key=lambda item: item[1], reverse=True)
    top = edges[:limit]
    if not top:
        return "(no edges)"
    return ", ".join(f"{other} ({weight:.1f})" for other, weight in top)


def build_edge_trace(graph, layout: dict[str, tuple[float, float]], top_k: int):
    """Optional faint line trace for the top_k highest-weight edges only."""
    ranked = sorted(
        graph.edges(data=True),
        key=lambda item: item[2].get("weight", 0.0),
        reverse=True,
    )[:top_k]

    xs: list[float | None] = []
    ys: list[float | None] = []
    for term_a, term_b, _ in ranked:
        if term_a not in layout or term_b not in layout:
            continue
        xs.extend([layout[term_a][0], layout[term_b][0], None])
        ys.extend([layout[term_a][1], layout[term_b][1], None])

    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="rgba(140,160,196,0.35)", width=1),
        hoverinfo="skip",
        showlegend=False,
        name="top edges",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive HTML layout preview.")
    parser.add_argument(
        "--show-edges",
        type=int,
        default=0,
        metavar="N",
        help="Draw the N highest-weight edges as faint lines (default: 0, off). "
        "Drawing all edges on a large graph makes the plot slow and cluttered, "
        "so this is opt-in and capped.",
    )
    args = parser.parse_args()

    if not GRAPH_PATH.is_file():
        print(f"ERROR: Missing input file: {GRAPH_PATH}", file=sys.stderr)
        print("Run scripts/run_graph_build.py first.", file=sys.stderr)
        return 1
    if not LAYOUT_JSON_PATH.is_file():
        print(f"ERROR: Missing input file: {LAYOUT_JSON_PATH}", file=sys.stderr)
        print("Run scripts/run_layout.py first.", file=sys.stderr)
        return 1

    graph = load_graph(GRAPH_PATH)
    layout = load_layout(LAYOUT_JSON_PATH)

    terms = [term for term in graph.nodes if term in layout]
    xs = [layout[term][0] for term in terms]
    ys = [layout[term][1] for term in terms]
    chat_counts = [graph.nodes[term].get("chat_count", 0) for term in terms]
    sizes = scaled_marker_sizes(chat_counts)

    hover_text = [
        f"<b>{term}</b><br>chat_count: {count}<br>top neighbors: {top_neighbors(graph, term)}"
        for term, count in zip(terms, chat_counts)
    ]

    fig = go.Figure()

    if args.show_edges > 0:
        fig.add_trace(build_edge_trace(graph, layout, args.show_edges))

    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=sizes,
                color=chat_counts,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="chat_count"),
                line=dict(width=0),
                opacity=0.85,
            ),
            text=hover_text,
            hoverinfo="text",
            name="keywords",
        )
    )

    fig.update_layout(
        title=f"Keyword co-occurrence layout — {len(terms)} nodes (scroll to zoom, drag to pan)",
        xaxis_title="x",
        yaxis_title="y",
        width=1400,
        height=1400,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    OUTPUT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_HTML_PATH), include_plotlyjs=True, full_html=True)

    print("Interactive layout preview")
    print("=" * 40)
    print(f"Nodes plotted: {len(terms)}")
    if args.show_edges > 0:
        print(f"Edges drawn (top by weight): {min(args.show_edges, graph.number_of_edges())}")
    print(f"Output: {OUTPUT_HTML_PATH.relative_to(PROJECT_ROOT)}")
    print("Open it directly in a browser (double-click, or `open` on macOS).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

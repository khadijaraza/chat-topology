#!/usr/bin/env python3
"""Project segment embeddings into 2D world coordinates (UMAP) and derive
conversation-concept / macro-domain centroid positions + segment heights.

Step 4 of the proximity-based 3-tier reconstruction (see CLAUDE.md). Run
after run_macro_domains.py. Output: data/processed/proximity_layout.json
and a static scatter preview colored by macro domain -- review the shape
before wiring build_output.py to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display backend needed
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conversation_concepts import load_segment_embeddings  # noqa: E402
from src.layout import get_coordinate_range  # noqa: E402
from src.proximity_layout import (  # noqa: E402
    apply_tangent_pull,
    compute_group_centroids,
    compute_segment_heights,
    compute_segment_positions,
    get_proximity_layout_config,
)

SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "segments.json"
SEGMENT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_embeddings.npz"
CONVERSATION_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "conversation_concepts.json"
MACRO_DOMAINS_PATH = PROJECT_ROOT / "data" / "processed" / "macro_domains.json"
PROXIMITY_LAYOUT_PATH = PROJECT_ROOT / "data" / "processed" / "proximity_layout.json"
PREVIEW_PNG_PATH = PROJECT_ROOT / "data" / "processed" / "proximity_layout_preview.png"

DOMAIN_COLORS = [
    "#e94560", "#0f8b8d", "#f9a826", "#6a4c93", "#1a936f", "#c44536",
    "#3a86ff", "#ff6392", "#7d8570", "#ffbe0b", "#8338ec", "#3d5a80",
]


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (SEGMENTS_PATH, "scripts/run_segmentation.py"),
            (SEGMENT_EMBEDDINGS_PATH, "scripts/run_grounded_extraction.py"),
            (CONVERSATION_CONCEPTS_PATH, "scripts/run_conversation_concepts.py"),
            (MACRO_DOMAINS_PATH, "scripts/run_macro_domains.py"),
        ]
        if not path.is_file()
    ]
    if missing:
        for path, script in missing:
            print(f"ERROR: Missing input file: {path}", file=sys.stderr)
            print(f"Run {script} first.", file=sys.stderr)
        return 1

    with SEGMENTS_PATH.open(encoding="utf-8") as handle:
        segments_by_chat = json.load(handle)
    with CONVERSATION_CONCEPTS_PATH.open(encoding="utf-8") as handle:
        concept_summary = json.load(handle)
    with MACRO_DOMAINS_PATH.open(encoding="utf-8") as handle:
        domain_summary = json.load(handle)

    segment_ids, embeddings = load_segment_embeddings(SEGMENT_EMBEDDINGS_PATH)
    concept_to_segments = {label: info["segment_ids"] for label, info in concept_summary.items()}
    # macro_domains.json's "segment_ids" is authoritative (domains are now
    # clustered directly from segments, not derived from concept
    # membership -- see src/macro_domains.py) -- read it straight rather
    # than re-deriving via concepts, which could disagree when a concept's
    # segments split across domains.
    domain_to_segments = {label: info["segment_ids"] for label, info in domain_summary.items()}

    cfg = get_proximity_layout_config()
    print(
        f"UMAP-projecting {len(segment_ids)} segment embeddings "
        f"(n_neighbors={cfg['umap_n_neighbors']}, min_dist={cfg['umap_min_dist']}, "
        f"metric={cfg['umap_metric']})..."
    )
    segment_positions = compute_segment_positions(segment_ids, embeddings)

    if cfg["tangent_pull_alpha"] > 0:
        print(f"Applying tangent-pull (alpha={cfg['tangent_pull_alpha']})...")
        segment_positions = apply_tangent_pull(segment_positions, segments_by_chat, cfg["tangent_pull_alpha"])

    concept_positions = compute_group_centroids(concept_to_segments, segment_positions)
    domain_positions = compute_group_centroids(domain_to_segments, segment_positions)

    segment_heights = compute_segment_heights(segments_by_chat)

    segment_to_concept = {s: label for label, members in concept_to_segments.items() for s in members}
    segment_to_domain = {s: label for label, segs in domain_to_segments.items() for s in segs}
    # macro_domains.json's "concepts" list is the majority-vote assignment
    # computed by assign_concepts_to_domains() (src/macro_domains.py) --
    # invert it once here rather than re-deriving per concept.
    concept_to_domain = {
        concept_label: domain_label
        for domain_label, info in domain_summary.items()
        for concept_label in info.get("concepts", [])
    }

    payload = {
        "coordinate_range": get_coordinate_range(),
        "segments": {
            segment_id: {
                "x": segment_positions[segment_id][0],
                "y": segment_positions[segment_id][1],
                "z": segment_heights.get(segment_id, 0.0),
                "conversation_concept": segment_to_concept.get(segment_id),
                "macro_domain": segment_to_domain.get(segment_id),
            }
            for segment_id in segment_positions
        },
        "conversation_concepts": {
            label: {"x": pos[0], "y": pos[1], "macro_domain": concept_to_domain.get(label)}
            for label, pos in concept_positions.items()
        },
        "macro_domains": {label: {"x": pos[0], "y": pos[1]} for label, pos in domain_positions.items()},
    }

    PROXIMITY_LAYOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROXIMITY_LAYOUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # Preview: every segment, colored by macro domain, plus labeled domain centroids.
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    domain_labels = sorted(domain_positions, key=lambda d: -len(domain_to_segments.get(d, [])))
    color_by_domain = {label: DOMAIN_COLORS[i % len(DOMAIN_COLORS)] for i, label in enumerate(domain_labels)}

    for domain_label in domain_labels:
        seg_ids = [s for s in domain_to_segments.get(domain_label, []) if s in segment_positions]
        xs = [segment_positions[s][0] for s in seg_ids]
        ys = [segment_positions[s][1] for s in seg_ids]
        ax.scatter(xs, ys, s=12, c=color_by_domain[domain_label], alpha=0.55, linewidths=0, zorder=1)

    for domain_label, (x, y) in domain_positions.items():
        ax.scatter([x], [y], s=140, c=color_by_domain[domain_label], edgecolors="black", linewidths=1.2, zorder=3)
        ax.annotate(
            domain_label, (x, y), fontsize=9, fontweight="bold", color="#1a1a2e",
            xytext=(5, 5), textcoords="offset points", zorder=4,
        )

    coordinate_range = get_coordinate_range()
    margin = coordinate_range * 1.15
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.set_title(f"Proximity layout preview ({len(segment_positions)} segments, {len(domain_positions)} macro domains)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(PREVIEW_PNG_PATH)
    plt.close(fig)

    print()
    print("Proximity layout summary")
    print("=" * 40)
    print(f"Segments positioned: {len(segment_positions)}")
    print(f"Conversation concepts positioned: {len(concept_positions)}")
    print(f"Macro domains positioned: {len(domain_positions)}")
    print(f"Coordinate range: [-{coordinate_range:g}, {coordinate_range:g}] (config/layout.yaml)")
    print(f"Output: {PROXIMITY_LAYOUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Preview PNG: {PREVIEW_PNG_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print("Macro domain positions (centroid of member segments):")
    for label in domain_labels:
        x, y = domain_positions[label]
        n = len(domain_to_segments.get(label, []))
        print(f"  {label:<30s} ({x:7.2f}, {y:7.2f})  segments={n}")

    print()
    print("BEFORE YOU MOVE ON: open the preview PNG -- do same-domain segments")
    print("visually cluster together, and do domain centroids sit apart from each")
    print("other rather than piling up in the middle? Tune umap_n_neighbors/")
    print("umap_min_dist in config/proximity_layout.yaml and rerun if the shape")
    print("looks collapsed or scrambled.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

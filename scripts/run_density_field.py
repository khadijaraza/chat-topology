#!/usr/bin/env python3
"""Fit a Dirichlet Process Gaussian Mixture Model (DPGMM) over the
proximity layout's segment positions and evaluate its density on a grid --
a continuous, alternative reading of the corpus to the discrete Macro
Domains tier.

Run this after run_proximity_layout.py. Output,
data/processed/density_field.json, holds the evaluation grid + log-density
values the frontend renders as a topographic terrain mesh underneath the
point cloud. Also saves a static contour preview PNG -- review it before
wiring the frontend to this, same "before you move on" discipline as every
other clustering/layout step in this pipeline.
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

from src.density_field import (  # noqa: E402
    effective_component_count,
    evaluate_density_grid,
    fit_dpgmm,
    get_density_field_config,
)
from src.layout import get_coordinate_range  # noqa: E402

PROXIMITY_LAYOUT_PATH = PROJECT_ROOT / "data" / "processed" / "proximity_layout.json"
MACRO_DOMAINS_PATH = PROJECT_ROOT / "data" / "processed" / "macro_domains.json"
DENSITY_FIELD_PATH = PROJECT_ROOT / "data" / "processed" / "density_field.json"
PREVIEW_PNG_PATH = PROJECT_ROOT / "data" / "processed" / "density_field_preview.png"


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (PROXIMITY_LAYOUT_PATH, "scripts/run_proximity_layout.py"),
            (MACRO_DOMAINS_PATH, "scripts/run_macro_domains.py"),
        ]
        if not path.is_file()
    ]
    if missing:
        for path, script in missing:
            print(f"ERROR: Missing input file: {path}", file=sys.stderr)
            print(f"Run {script} first.", file=sys.stderr)
        return 1

    with PROXIMITY_LAYOUT_PATH.open(encoding="utf-8") as handle:
        layout = json.load(handle)
    with MACRO_DOMAINS_PATH.open(encoding="utf-8") as handle:
        domain_summary = json.load(handle)

    segment_positions: dict[str, dict] = layout["segments"]
    segment_ids = list(segment_positions)
    positions = np.array([[segment_positions[s]["x"], segment_positions[s]["y"]] for s in segment_ids])

    cfg = get_density_field_config()
    print(
        f"Fitting DPGMM over {len(segment_ids)} segment positions "
        f"(max_components={cfg['max_components']}, "
        f"weight_concentration_prior={cfg['weight_concentration_prior']}, "
        f"covariance_type={cfg['covariance_type']})..."
    )
    model = fit_dpgmm(positions)
    effective = effective_component_count(model)
    print(f"Effective components kept by the Dirichlet process prior: {effective} "
          f"(of {cfg['max_components']} candidates)")

    coordinate_range = get_coordinate_range()
    print(f"Evaluating density on a {cfg['grid_resolution']}x{cfg['grid_resolution']} grid "
          f"over [-{coordinate_range:g}, {coordinate_range:g}]...")
    grid = evaluate_density_grid(model, coordinate_range, cfg["grid_resolution"])
    grid["terrain_z_range"] = cfg["terrain_z_range"]

    DENSITY_FIELD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DENSITY_FIELD_PATH.open("w", encoding="utf-8") as handle:
        json.dump(grid, handle)
        handle.write("\n")

    # Preview: filled contour of log-density, segment points colored by
    # macro domain overlaid on top, so the real clusters can be checked
    # against where the terrain actually puts its "mountains."
    domain_by_segment: dict[str, str] = {}
    for label, info in domain_summary.items():
        for segment_id in info["segment_ids"]:
            domain_by_segment[segment_id] = label

    xs = np.array(grid["x"])
    ys = np.array(grid["y"])
    normalized_density = np.array(grid["normalized_density"])

    # Rendered from normalized_density (percentile rank), not raw
    # log_density -- see evaluate_density_grid()'s docstring: a few tight,
    # high-precision components produce spikes hundreds of nats taller
    # than the ridges between components, which saturates a raw (even
    # percentile-clipped) log_density color scale to "maximum" almost
    # everywhere. The percentile-rank transform is what actually makes the
    # real multi-cluster texture visible.
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    contour = ax.contourf(xs, ys, normalized_density, levels=np.linspace(0, 1, 30), cmap="viridis")
    fig.colorbar(contour, ax=ax, label="normalized density (percentile rank)", fraction=0.046, pad=0.04)

    point_xs = [segment_positions[s]["x"] for s in segment_ids]
    point_ys = [segment_positions[s]["y"] for s in segment_ids]
    ax.scatter(point_xs, point_ys, s=4, c="white", alpha=0.5, linewidths=0, zorder=2)

    margin = coordinate_range * 1.15
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal")
    ax.set_title(f"Density field preview ({effective} effective DPGMM components, "
                 f"{len(segment_ids)} segments)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(PREVIEW_PNG_PATH)
    plt.close(fig)

    print()
    print("Density field summary")
    print("=" * 40)
    log_density_arr = np.array(grid["log_density"])
    print(f"Grid: {cfg['grid_resolution']}x{cfg['grid_resolution']}")
    print(f"log-density range: [{log_density_arr.min():.2f}, {log_density_arr.max():.2f}] "
          f"(rendered via percentile-rank normalization, see evaluate_density_grid() docstring)")
    print(f"Output: {DENSITY_FIELD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Preview PNG: {PREVIEW_PNG_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("BEFORE YOU MOVE ON: open the preview PNG -- do the bright ridges (\"mountains\")")
    print("line up with where the real, dense Macro Domains actually are, and do sparse")
    print("segments (Background / one-offs) sit in the darker \"lowland\" areas? If the")
    print("terrain looks like one or two spikes on a flat plain, or looks like pure noise")
    print("with no real structure, tune weight_concentration_prior/max_components in")
    print("config/density_field.yaml and rerun.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

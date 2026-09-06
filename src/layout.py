"""Compute a 2D force-directed layout for the keyword co-occurrence graph."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import networkx as nx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAYOUT_CONFIG_PATH = PROJECT_ROOT / "config" / "layout.yaml"

# Fallback used only if config/layout.yaml is missing or malformed.
FALLBACK_COORDINATE_RANGE = 100.0


@lru_cache(maxsize=1)
def get_coordinate_range(path: str = str(DEFAULT_LAYOUT_CONFIG_PATH)) -> float:
    """Load the world-coordinate range from config/layout.yaml.

    This is the single source of truth for how far layout coordinates can
    spread on either axis (see the comment in that file) — the frontend's
    3D scene bounds are expected to match it.
    """
    config_path = Path(path)
    if not config_path.is_file():
        return FALLBACK_COORDINATE_RANGE
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    value = payload.get("coordinate_range", FALLBACK_COORDINATE_RANGE)
    try:
        return float(value)
    except (TypeError, ValueError):
        return FALLBACK_COORDINATE_RANGE


def normalize_positions(
    pos: dict[str, tuple[float, float]],
    *,
    coordinate_range: float,
) -> dict[str, tuple[float, float]]:
    """Rescale positions to fit within [-coordinate_range, +coordinate_range].

    Scales both axes by the same factor (derived from whichever axis has the
    larger extent) so relative distances — and therefore cluster shape — are
    preserved rather than stretched independently per axis.
    """
    if not pos:
        return {}

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    half_extent = max(x_max - x_min, y_max - y_min) / 2
    if half_extent <= 0:
        # All nodes coincide (e.g. a single node, or a fully degenerate
        # layout) — place everything at the origin rather than dividing by
        # zero.
        return {term: (0.0, 0.0) for term in pos}

    scale = coordinate_range / half_extent
    return {
        term: ((x - center_x) * scale, (y - center_y) * scale)
        for term, (x, y) in pos.items()
    }


def compute_layout(
    graph: nx.Graph,
    *,
    seed: int = 42,
    iterations: int = 50,
) -> dict[str, tuple[float, float]]:
    """Compute a 2D force-directed layout, normalized to world coordinates.

    Uses networkx.spring_layout with edge ``weight`` so more-connected /
    higher-cooccurrence keyword pairs are pulled closer together. Output
    coordinates are normalized to roughly [-coordinate_range, coordinate_range]
    on both axes (see config/layout.yaml), suitable for direct use as world
    coordinates in a 3D scene.

    Returns an empty dict for an empty graph.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_nodes() == 1:
        (only_node,) = graph.nodes
        return {only_node: (0.0, 0.0)}

    raw_pos = nx.spring_layout(
        graph,
        weight="weight",
        seed=seed,
        iterations=iterations,
        dim=2,
    )

    coordinate_range = get_coordinate_range()
    return normalize_positions(raw_pos, coordinate_range=coordinate_range)
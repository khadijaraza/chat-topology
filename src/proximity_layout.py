"""Project segment embeddings into 2D/3D world coordinates for the
proximity-based visualization (segments -> conversation concepts -> macro
domains). Step 4 of the pipeline reconstruction -- see CLAUDE.md.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.height_score import get_height_config
from src.layout import get_coordinate_range, normalize_positions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROXIMITY_LAYOUT_CONFIG_PATH = PROJECT_ROOT / "config" / "proximity_layout.yaml"

# Fallbacks used only if config/proximity_layout.yaml is missing or malformed.
FALLBACK_UMAP_N_NEIGHBORS = 15
FALLBACK_UMAP_MIN_DIST = 0.1
FALLBACK_UMAP_METRIC = "cosine"
FALLBACK_TANGENT_PULL_ALPHA = 0.15


@lru_cache(maxsize=1)
def get_proximity_layout_config(
    path: str = str(DEFAULT_PROXIMITY_LAYOUT_CONFIG_PATH),
) -> dict:
    """Load UMAP + tangent-pull settings from config/proximity_layout.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {
            "umap_n_neighbors": FALLBACK_UMAP_N_NEIGHBORS,
            "umap_min_dist": FALLBACK_UMAP_MIN_DIST,
            "umap_metric": FALLBACK_UMAP_METRIC,
            "tangent_pull_alpha": FALLBACK_TANGENT_PULL_ALPHA,
        }

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "umap_n_neighbors": int(payload.get("umap_n_neighbors", FALLBACK_UMAP_N_NEIGHBORS)),
        "umap_min_dist": float(payload.get("umap_min_dist", FALLBACK_UMAP_MIN_DIST)),
        "umap_metric": payload.get("umap_metric", FALLBACK_UMAP_METRIC),
        "tangent_pull_alpha": float(payload.get("tangent_pull_alpha", FALLBACK_TANGENT_PULL_ALPHA)),
    }


def project_to_2d(segment_ids: list[str], embeddings: np.ndarray) -> dict[str, tuple[float, float]]:
    """UMAP-project segment embeddings into raw (unnormalized) 2D coordinates.

    Replaces the PMI co-occurrence graph + spring layout (src/layout.py) for
    this pathway -- positions come directly from embedding-space proximity,
    not from any co-occurrence signal.
    """
    if not segment_ids:
        return {}
    if len(segment_ids) == 1:
        return {segment_ids[0]: (0.0, 0.0)}

    import umap

    cfg = get_proximity_layout_config()
    # n_neighbors can't exceed n_samples - 1 (umap-learn's own constraint);
    # clamp rather than error on small/synthetic inputs.
    n_neighbors = min(cfg["umap_n_neighbors"], len(segment_ids) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=cfg["umap_min_dist"],
        metric=cfg["umap_metric"],
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    return {segment_id: (float(x), float(y)) for segment_id, (x, y) in zip(segment_ids, coords)}


def compute_segment_positions(
    segment_ids: list[str], embeddings: np.ndarray
) -> dict[str, tuple[float, float]]:
    """UMAP-project then rescale to the established [-coordinate_range,
    +coordinate_range] world-coordinate contract (config/layout.yaml)."""
    raw_positions = project_to_2d(segment_ids, embeddings)
    return normalize_positions(raw_positions, coordinate_range=get_coordinate_range())


def apply_tangent_pull(
    positions: dict[str, tuple[float, float]],
    segments_by_chat: dict[str, list[dict]],
    alpha: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Blend each "tangent" segment's position toward its own chat's "core"
    segment centroid: P_final = (1 - alpha) * P_tangent + alpha * P_core.

    alpha=0 (or no core segments in that chat) leaves a segment's position
    untouched. "core" segments are themselves never pulled.
    """
    if alpha is None:
        alpha = get_proximity_layout_config()["tangent_pull_alpha"]
    if alpha <= 0:
        return dict(positions)

    result = dict(positions)
    for chat_id, segments in segments_by_chat.items():
        core_positions = [
            positions[seg["segment_id"]]
            for seg in segments
            if seg.get("label") == "core" and seg["segment_id"] in positions
        ]
        if not core_positions:
            continue
        core_x = sum(p[0] for p in core_positions) / len(core_positions)
        core_y = sum(p[1] for p in core_positions) / len(core_positions)

        for seg in segments:
            if seg.get("label") != "tangent":
                continue
            segment_id = seg["segment_id"]
            if segment_id not in positions:
                continue
            x, y = positions[segment_id]
            result[segment_id] = ((1 - alpha) * x + alpha * core_x, (1 - alpha) * y + alpha * core_y)

    return result


def compute_group_centroids(
    group_to_members: dict[str, list[str]],
    member_positions: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Centroid (mean position) of each group's member positions -- shared by
    both conversation-concept and macro-domain positioning, guaranteeing
    visual continuity across zoom levels (same pattern already proven for
    supertopic centroids in the word-level pipeline)."""
    centroids: dict[str, tuple[float, float]] = {}
    for label, members in group_to_members.items():
        pts = [member_positions[m] for m in members if m in member_positions]
        if not pts:
            continue
        centroids[label] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return centroids


def _min_max_scale(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    spread = hi - lo
    if spread <= 0:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / spread for key, value in values.items()}


def compute_segment_heights(segments_by_chat: dict[str, list[dict]]) -> dict[str, float]:
    """Z-axis height per segment: log1p(word_count), min-max scaled to
    [0, z_range] (config/height_weights.yaml, shared contract with the
    word-level pipeline's height scores -- same log-scale reasoning applies,
    segment word counts are heavily right-skewed too).

    Deliberate simplification vs. the word-level height_score.py: a segment
    belongs to exactly one chat, so there's no cross-chat frequency/
    sustained-focus signal to combine -- just its own word count.
    """
    _, z_range = get_height_config()
    raw = {
        seg["segment_id"]: math.log1p(seg.get("word_count", 0))
        for segments in segments_by_chat.values()
        for seg in segments
    }
    normalized = _min_max_scale(raw)
    return {segment_id: value * z_range for segment_id, value in normalized.items()}

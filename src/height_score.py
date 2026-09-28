"""Compute per-topic height (Z-axis) scores from chat engagement depth."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import networkx as nx
import yaml

from src.build_graph import terms_in_chat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEIGHT_CONFIG_PATH = PROJECT_ROOT / "config" / "height_weights.yaml"

# Fallbacks used only if config/height_weights.yaml is missing or malformed.
FALLBACK_WEIGHTS = {"frequency": 0.4, "depth": 0.3, "sustained_focus": 0.3}
FALLBACK_Z_RANGE = 50.0


@lru_cache(maxsize=1)
def get_height_config(
    path: str = str(DEFAULT_HEIGHT_CONFIG_PATH),
) -> tuple[dict[str, float], float]:
    """Load height-score weights and the Z-axis range from config/height_weights.yaml.

    This is the single source of truth for how the three sub-metrics are
    weighted and how far the combined score spreads on the Z axis -- see the
    comments in that file for how it pairs with config/layout.yaml's
    coordinate_range.
    """
    config_path = Path(path)
    if not config_path.is_file():
        return dict(FALLBACK_WEIGHTS), FALLBACK_Z_RANGE

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    raw_weights = payload.get("weights", {})
    if not isinstance(raw_weights, dict):
        raw_weights = {}

    weights: dict[str, float] = {}
    for key, default in FALLBACK_WEIGHTS.items():
        try:
            weights[key] = float(raw_weights.get(key, default))
        except (TypeError, ValueError):
            weights[key] = default

    try:
        z_range = float(payload.get("z_range", FALLBACK_Z_RANGE))
    except (TypeError, ValueError):
        z_range = FALLBACK_Z_RANGE

    return weights, z_range


def _chat_word_count(conversation: dict) -> int:
    """Word count across every turn's text (prompts and responses alike)."""
    total = 0
    for turn in conversation.get("turns", []):
        if not isinstance(turn, dict):
            continue
        text = turn.get("text")
        if isinstance(text, str):
            total += len(text.split())
    return total


def _min_max_scale(values: dict[str, float]) -> dict[str, float]:
    """Scale values to [0, 1]. A constant (zero-spread) input maps to all 0.0."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    spread = hi - lo
    if spread <= 0:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / spread for key, value in values.items()}


def compute_height_scores(
    graph: nx.Graph,
    conversations: list[dict],
    chat_keywords: list[dict],
) -> dict[str, float]:
    """Compute a [0, z_range] height (Z) score for every node in ``graph``.

    For each topic, three raw metrics are computed from the chats that
    contain it:
      - metric_frequency: distinct chat count (the ``chat_count`` node
        attribute already computed by build_cooccurrence_graph)
      - metric_depth: total word count (prompts + responses) summed across
        all of those chats
      - metric_sustained_focus: word count of the single longest chat among
        them (not summed -- one long thread, not many short ones)

    Each raw metric is log1p-normalized independently -- these are heavily
    right-skewed distributions, and without log-scaling one mega-thread would
    produce a single skyscraper next to an otherwise flat plain -- then
    min-max scaled to [0, 1]. The three normalized metrics are combined via a
    weighted sum (weights/self-normalized-by-their-sum from
    config/height_weights.yaml, so hand-edited weights don't need to add to
    exactly 1.0) and the result is scaled to [0, z_range] (also in that
    file).

    Returns an empty dict for an empty graph.
    """
    if graph.number_of_nodes() == 0:
        return {}

    word_counts: dict[str, int] = {}
    for conversation in conversations:
        chat_id = conversation.get("chat_id")
        if isinstance(chat_id, str):
            word_counts[chat_id] = _chat_word_count(conversation)

    chats_by_term: dict[str, set[str]] = {}
    for entry in chat_keywords:
        chat_id = entry.get("chat_id")
        if not isinstance(chat_id, str):
            continue
        for term in terms_in_chat(entry):
            chats_by_term.setdefault(term, set()).add(chat_id)

    raw_frequency: dict[str, float] = {}
    raw_depth: dict[str, float] = {}
    raw_sustained_focus: dict[str, float] = {}

    for term, attrs in graph.nodes(data=True):
        chat_ids = chats_by_term.get(term, set())
        counts_for_term = [word_counts.get(chat_id, 0) for chat_id in chat_ids]

        raw_frequency[term] = float(attrs.get("chat_count", len(chat_ids)))
        raw_depth[term] = float(sum(counts_for_term))
        raw_sustained_focus[term] = float(max(counts_for_term, default=0))

    norm_frequency = _min_max_scale({t: math.log1p(v) for t, v in raw_frequency.items()})
    norm_depth = _min_max_scale({t: math.log1p(v) for t, v in raw_depth.items()})
    norm_sustained_focus = _min_max_scale(
        {t: math.log1p(v) for t, v in raw_sustained_focus.items()}
    )

    weights, z_range = get_height_config()
    weight_total = sum(weights.values()) or 1.0

    scores: dict[str, float] = {}
    for term in graph.nodes:
        combined = (
            weights["frequency"] * norm_frequency.get(term, 0.0)
            + weights["depth"] * norm_depth.get(term, 0.0)
            + weights["sustained_focus"] * norm_sustained_focus.get(term, 0.0)
        ) / weight_total
        scores[term] = combined * z_range

    return scores

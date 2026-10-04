"""Split each chat into discourse-cohesive segments before extraction.

A chat that drifts through several unrelated topics gets treated as one
undifferentiated bag of words by whole-chat extraction, creating false
co-occurrence between things that were never really "together" (e.g. a C
allocation bug and, three turns later, flight baggage allowances). This
module detects both abrupt topic changes (a sharp similarity drop between
adjacent windows -- TextTiling-style valley detection) and gradual drift
(a slow decay away from the opening topic that stays down) and splits the
chat into "core" (largest) and "tangent" (everything else) segments.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.text_embedding import embed_long_text, get_embedding_model, mean_pool_normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEGMENTATION_CONFIG_PATH = PROJECT_ROOT / "config" / "segmentation.yaml"

# Fallbacks used only if config/segmentation.yaml is missing or malformed.
FALLBACK_CONFIG = {
    "model_name": "all-mpnet-base-v2",
    "window_turns": 2,
    "min_turns_for_segmentation": 8,
    "max_embedding_tokens": 384,
    "hard_shift_lookback": 3,
    "hard_shift_z": 1.0,
    "hard_shift_max_similarity": 0.4,
    "drift_tolerance": 0.15,
    "drift_min_consecutive": 3,
}

@lru_cache(maxsize=1)
def get_segmentation_config(path: str = str(DEFAULT_SEGMENTATION_CONFIG_PATH)) -> dict:
    """Load window/threshold settings from config/segmentation.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return dict(FALLBACK_CONFIG)

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {key: payload.get(key, default) for key, default in FALLBACK_CONFIG.items()}


@dataclass
class Window:
    start_turn: int
    end_turn: int  # exclusive
    text: str


def build_windows(turns: list[dict], window_turns: int) -> list[Window]:
    """Slice a chat's turns into overlapping windows, stride 1."""
    n = len(turns)
    if n < window_turns:
        return []
    windows = []
    for start in range(0, n - window_turns + 1):
        end = start + window_turns
        text = "\n".join(
            turn["text"] for turn in turns[start:end] if isinstance(turn.get("text"), str)
        )
        windows.append(Window(start_turn=start, end_turn=end, text=text))
    return windows


def embed_window(text: str, model, max_tokens: int) -> np.ndarray:
    """Embed a window's text -- thin wrapper over the shared chunk+pool
    embedder (src/text_embedding.py), kept for naming clarity at call sites."""
    return embed_long_text(text, model, max_tokens)


def similarity_trajectory(window_embeddings: list[np.ndarray]) -> list[float]:
    """Adjacent-window cosine similarities (embeddings are unit-normalized,
    so this is just the dot product)."""
    return [
        float(np.dot(window_embeddings[i], window_embeddings[i + 1]))
        for i in range(len(window_embeddings) - 1)
    ]


def detect_hard_shifts(
    trajectory: list[float], lookback: int, z: float, max_similarity: float
) -> list[int]:
    """TextTiling-style valley detection: adjacent-window boundary index of
    each hard shift. A boundary's depth score is (local peak before it -
    its own similarity) + (local peak after it - its own similarity).

    A boundary counts as a shift only if BOTH its depth exceeds mean +
    z*std for this chat (adaptive -- relative to how noisy this chat's own
    trajectory is) AND its raw similarity is below max_similarity (an
    absolute floor). z alone isn't enough: on a long chat, many windows
    end up locally lower than their neighbors just from natural sub-topic
    variation within one coherent task, not real tangents -- see
    config/segmentation.yaml for the real-data numbers that motivated this.
    """
    n = len(trajectory)
    if n < 3:
        return []

    depth_scores = []
    for b in range(n):
        left = trajectory[max(0, b - lookback):b]
        right = trajectory[b + 1:b + 1 + lookback]
        left_peak = max(left) if left else trajectory[b]
        right_peak = max(right) if right else trajectory[b]
        depth_scores.append((left_peak - trajectory[b]) + (right_peak - trajectory[b]))

    mean_depth = statistics.mean(depth_scores)
    std_depth = statistics.pstdev(depth_scores)
    threshold = mean_depth + z * std_depth

    return [
        b
        for b, depth in enumerate(depth_scores)
        if depth > threshold and depth > 0 and trajectory[b] < max_similarity
    ]


def detect_drift(
    window_embeddings: list[np.ndarray], tolerance: float, min_consecutive: int
) -> int | None:
    """Rolling similarity to the initial segment's centroid (mean of the
    first min_consecutive windows). Returns the window index where
    sustained divergence begins, or None if it never sustains."""
    if len(window_embeddings) < min_consecutive + 1:
        return None

    anchor = mean_pool_normalize(np.stack(window_embeddings[:min_consecutive]))
    threshold = 1.0 - tolerance

    consecutive = 0
    for i, embedding in enumerate(window_embeddings):
        similarity = float(np.dot(embedding, anchor))
        if similarity < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                return i - min_consecutive + 1
        else:
            consecutive = 0
    return None


def _make_segment(
    chat_id: str,
    seg_index: int,
    turns_slice: list[dict],
    start_turn: int,
    end_turn: int,
    boundary_type: str | None,
) -> dict:
    word_count = sum(
        len(turn["text"].split()) for turn in turns_slice if isinstance(turn.get("text"), str)
    )
    return {
        "segment_id": f"{chat_id}::seg{seg_index}",
        "chat_id": chat_id,
        "start_turn": start_turn,
        "end_turn": end_turn,
        "turn_count": end_turn - start_turn,
        "word_count": word_count,
        "boundary_type": boundary_type,
        "label": "core",  # overwritten by _label_segments()
    }


def _label_segments(segments: list[dict]) -> None:
    """The largest segment (by word count) is "core"; everything else is
    "tangent" -- matches the plan's definition of Core Thread vs. Tangent
    Branch."""
    if not segments:
        return
    largest = max(segments, key=lambda seg: seg["word_count"])
    for segment in segments:
        segment["label"] = "core" if segment is largest else "tangent"


def segment_chat(conversation: dict, config: dict, model) -> list[dict]:
    """Segment one chat. Chats under min_turns_for_segmentation short-
    circuit to a single "core" segment spanning the whole chat -- not
    enough length for a meaningful similarity trajectory."""
    turns = conversation.get("turns", [])
    chat_id = conversation.get("chat_id", "")
    n_turns = len(turns)

    if n_turns < config["min_turns_for_segmentation"]:
        return [_make_segment(chat_id, 0, turns, 0, n_turns, boundary_type=None)]

    windows = build_windows(turns, config["window_turns"])
    if len(windows) < 2:
        return [_make_segment(chat_id, 0, turns, 0, n_turns, boundary_type=None)]

    embeddings = [embed_window(w.text, model, config["max_embedding_tokens"]) for w in windows]
    trajectory = similarity_trajectory(embeddings)

    hard_shifts = detect_hard_shifts(
        trajectory, config["hard_shift_lookback"], config["hard_shift_z"], config["hard_shift_max_similarity"]
    )
    drift_start = detect_drift(embeddings, config["drift_tolerance"], config["drift_min_consecutive"])

    boundary_types: dict[int, str] = {}
    for boundary in hard_shifts:
        turn = boundary + 1  # window (boundary+1) starts here, stride 1
        boundary_types[turn] = "hard_shift"
    if drift_start is not None and drift_start not in boundary_types:
        boundary_types[drift_start] = "drift"

    boundary_turns = sorted(turn for turn in boundary_types if 0 < turn < n_turns)

    segments = []
    starts = [0] + boundary_turns
    ends = boundary_turns + [n_turns]
    for seg_index, (start, end) in enumerate(zip(starts, ends)):
        boundary_type = boundary_types.get(start) if start > 0 else None
        segments.append(_make_segment(chat_id, seg_index, turns[start:end], start, end, boundary_type))

    _label_segments(segments)
    return segments


def segment_conversations(conversations: list[dict]) -> dict[str, list[dict]]:
    """Segment every conversation. Returns {chat_id: [segment, ...]}."""
    config = get_segmentation_config()
    model = get_embedding_model(config["model_name"])

    result: dict[str, list[dict]] = {}
    for conversation in conversations:
        chat_id = conversation.get("chat_id")
        if not isinstance(chat_id, str):
            continue
        result[chat_id] = segment_chat(conversation, config, model)
    return result

"""Group segments into conversation-level concepts via embedding clustering.

Mid tier of the proximity visualization's 3-layer hierarchy (segments ->
conversation concepts -> macro domains -- see CLAUDE.md). Unlike the
original word-level concept_clustering.py (which clusters individual
keyword embeddings), this clusters whole SEGMENT embeddings directly, so
two segments end up in the same concept because their overall content is
similar -- e.g. every differential-equations segment across many different
chats collapses into one "Linear Systems & ODEs" neighborhood, regardless
of which specific keywords each one happened to surface.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONVERSATION_CONCEPTS_CONFIG_PATH = PROJECT_ROOT / "config" / "conversation_concepts.yaml"

# Fallbacks used only if config/conversation_concepts.yaml is missing or malformed.
FALLBACK_MIN_CLUSTER_SIZE = 3
FALLBACK_MIN_SAMPLES = 4

# HDBSCAN's noise label for points it declines to assign to any cluster.
NOISE_LABEL = -1


@lru_cache(maxsize=1)
def get_conversation_concepts_config(path: str = str(DEFAULT_CONVERSATION_CONCEPTS_CONFIG_PATH)) -> dict:
    """Load HDBSCAN settings from config/conversation_concepts.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {"min_cluster_size": FALLBACK_MIN_CLUSTER_SIZE, "min_samples": FALLBACK_MIN_SAMPLES}

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "min_cluster_size": int(payload.get("min_cluster_size", FALLBACK_MIN_CLUSTER_SIZE)),
        "min_samples": payload.get("min_samples", FALLBACK_MIN_SAMPLES),
    }


def load_segment_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Load segment_embeddings.npz (see scripts/run_grounded_extraction.py).
    Returns (segment_ids, embeddings) with matching row order."""
    with np.load(path, allow_pickle=False) as data:
        segment_ids = [str(s) for s in data["segment_ids"]]
        embeddings = data["embeddings"]
    return segment_ids, embeddings


def cluster_segment_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """HDBSCAN-cluster segment embeddings. Returns one cluster id per row
    (-1 = noise). Same euclidean-on-unit-normalized-vectors reasoning as
    concept_clustering.cluster_terms() -- cosine similarity and euclidean
    distance are equivalent up to a monotonic transform here, and HDBSCAN
    doesn't support a 'cosine' metric directly."""
    import hdbscan

    cfg = get_conversation_concepts_config()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


def find_medoid(
    segment_ids: list[str],
    embeddings: np.ndarray,
    has_keyword: set[str] | None = None,
) -> str:
    """The member segment closest to the group's centroid -- used both to
    pick a representative label and (later, in the proximity layout) as a
    natural anchor point.

    Real bug found via manual frontend testing, not caught by the text-
    sample checkpoint: a handful of segments (the ones Phase 2 already
    documented as producing zero keyword candidates) have no entry in
    top_keyword_by_segment. When the pure geometric medoid happened to be
    one of those, cluster_segments_into_concepts() fell back to using the
    raw segment_id string ("43e85f9ad6284630::seg3") as the concept's
    LABEL -- an internal id leaking into a user-facing name. When
    ``has_keyword`` is given, the highest-centroid-similarity member that
    HAS a keyword is preferred; only if truly no member of the cluster has
    one does this fall back to the pure geometric medoid (that cluster's
    label then falls back further, see cluster_segments_into_concepts).
    """
    if len(segment_ids) == 1:
        return segment_ids[0]
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    similarities = embeddings @ centroid

    if has_keyword is not None:
        ranked = sorted(range(len(segment_ids)), key=lambda i: -similarities[i])
        for row in ranked:
            if segment_ids[row] in has_keyword:
                return segment_ids[row]

    return segment_ids[int(np.argmax(similarities))]


def cluster_segments_into_concepts(
    segment_ids: list[str],
    embeddings: np.ndarray,
    top_keyword_by_segment: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster every segment into a conversation concept.

    Each concept is labeled via its medoid segment's top keyword (already
    computed in segment_keywords.json -- cheaper and simpler than a fresh
    specificity-weighted pass over segment content).

    Returns (segment_id_to_concept_label, concept_label_to_member_segment_ids).
    Noise points (HDBSCAN couldn't group them with anything) each become
    their own singleton concept, so no segment's signal is silently
    dropped -- it just doesn't get merged with anything else.

    Labels are deduplicated before being used as the output dict's keys
    (numeric " (2)", " (3)", ... suffix on collision, same reasoning as
    src/build_output.py's _dedupe_slugs). Two DIFFERENT clusters can
    legitimately pick the same top keyword as their medoid's label (e.g.
    two unrelated singleton segments both surfacing "ucsd") -- a real bug
    here, found on real data: without deduping, the second cluster's
    dict write silently overwrote the first's entire member list, even
    though the cluster count itself (908) looked unchanged and its
    members were still correctly recorded in segment_to_concept. That
    mismatch -- concept_to_segments missing a segment that
    segment_to_concept still pointed at -- dropped 99 of 1,287 segments
    (7.7%) with no visible error, only caught by build_final_json_v3()'s
    schema validation downstream. min_cluster_size/min_samples in
    config/conversation_concepts.yaml are unaffected by this fix -- same
    cluster assignments, just no data loss when writing them out.
    """
    if not segment_ids:
        return {}, {}

    cluster_ids = cluster_segment_embeddings(embeddings)
    has_keyword = set(top_keyword_by_segment)

    members_by_cluster: dict[str, list[int]] = defaultdict(list)
    for row, cluster_id in enumerate(cluster_ids):
        key = f"__singleton__{segment_ids[row]}" if cluster_id == NOISE_LABEL else f"cluster_{cluster_id}"
        members_by_cluster[key].append(row)

    segment_to_concept: dict[str, str] = {}
    concept_to_segments: dict[str, list[str]] = {}
    seen_labels: set[str] = set()
    for cluster_key, rows in members_by_cluster.items():
        member_ids = [segment_ids[r] for r in rows]
        member_embeddings = embeddings[rows]
        medoid_id = find_medoid(member_ids, member_embeddings, has_keyword)
        # Only reached when NO member of this cluster has a keyword (every
        # member is one of the rare zero-candidate segments) -- a raw
        # segment_id is never used as a label, since that's an internal id,
        # not a human-readable term.
        base_label = top_keyword_by_segment.get(medoid_id, "(untitled)")

        label = base_label
        suffix = 2
        while label in seen_labels:
            label = f"{base_label} ({suffix})"
            suffix += 1
        seen_labels.add(label)

        concept_to_segments[label] = member_ids
        for segment_id in member_ids:
            segment_to_concept[segment_id] = label

    return segment_to_concept, concept_to_segments

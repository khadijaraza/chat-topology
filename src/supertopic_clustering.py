"""Group concept nodes into a small, fixed number of broader super-topics.

A second, coarser clustering pass on top of the concept layer
(src/concept_clustering.py): concept labels are embedded and grouped with
agglomerative clustering targeting a fixed, small cluster count (unlike
HDBSCAN's natural-count density clustering used for concepts), so the graph
has a legible, <100-node "zoomed out" view. Every concept is assigned to
exactly one super-topic -- there's no unclustered/noise bucket at this
layer, since a hard count cap and "some concepts left ungrouped" are in
tension (see config/supertopic_clustering.yaml).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.concept_clustering import embed_terms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUPERTOPIC_CONFIG_PATH = PROJECT_ROOT / "config" / "supertopic_clustering.yaml"

# Fallbacks used only if config/supertopic_clustering.yaml is missing or malformed.
FALLBACK_MODEL_NAME = "all-mpnet-base-v2"
FALLBACK_N_CLUSTERS = 80
FALLBACK_LINKAGE = "average"


@lru_cache(maxsize=1)
def get_supertopic_config(path: str = str(DEFAULT_SUPERTOPIC_CONFIG_PATH)) -> dict:
    """Load embedding/clustering settings from config/supertopic_clustering.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {
            "model_name": FALLBACK_MODEL_NAME,
            "n_clusters": FALLBACK_N_CLUSTERS,
            "linkage": FALLBACK_LINKAGE,
        }

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "model_name": payload.get("model_name", FALLBACK_MODEL_NAME),
        "n_clusters": int(payload.get("n_clusters", FALLBACK_N_CLUSTERS)),
        "linkage": payload.get("linkage", FALLBACK_LINKAGE),
    }


def cluster_concepts_into_supertopics(concept_labels: list[str]) -> np.ndarray:
    """Agglomeratively cluster concept label embeddings into a fixed count.

    Returns one cluster id per row; every concept gets assigned (no noise).
    """
    from sklearn.cluster import AgglomerativeClustering

    cfg = get_supertopic_config()
    embeddings = embed_terms(concept_labels, model_name=cfg["model_name"])
    n_clusters = min(cfg["n_clusters"], len(concept_labels))  # can't exceed point count
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=cfg["linkage"])
    return clusterer.fit_predict(embeddings)


def label_supertopics(
    supertopic_members: dict[int, list[str]], concept_chat_counts: Counter[str]
) -> dict[int, str]:
    """Pick each super-topic's most chat-frequent member concept as its label."""
    return {
        cluster_id: max(members, key=lambda label: (concept_chat_counts.get(label, 0), label))
        for cluster_id, members in supertopic_members.items()
    }


def cluster_into_supertopics(
    concept_labels: list[str], concept_chat_counts: Counter[str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster every concept label into a super-topic.

    Returns (concept_to_supertopic_label, supertopic_label_to_member_concepts).
    Every concept appears in exactly one super-topic's member list.
    """
    if not concept_labels:
        return {}, {}

    cluster_ids = cluster_concepts_into_supertopics(concept_labels)

    supertopic_members: dict[int, list[str]] = defaultdict(list)
    for label, cluster_id in zip(concept_labels, cluster_ids):
        supertopic_members[int(cluster_id)].append(label)

    supertopic_labels = label_supertopics(supertopic_members, concept_chat_counts)

    concept_to_supertopic: dict[str, str] = {}
    supertopic_to_concepts: dict[str, list[str]] = {}
    for cluster_id, members in supertopic_members.items():
        label = supertopic_labels[cluster_id]
        supertopic_to_concepts[label] = members
        for concept_label in members:
            concept_to_supertopic[concept_label] = label

    return concept_to_supertopic, supertopic_to_concepts

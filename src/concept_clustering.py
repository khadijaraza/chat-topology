"""Group extracted keywords into semantic concepts via embedding clustering.

Concepts, not raw keywords, become the graph's nodes: two keywords end up
in the same concept because they mean similar things (per sentence
embeddings), independent of whether they ever co-occur in a chat. Concept-
to-concept *edges* still come from co-occurrence -- see
build_chat_concepts(), which re-keys each chat's keyword hits by concept so
the existing co-occurrence graph builder (src/build_graph.py) can run on
concepts unmodified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.build_graph import compute_chat_counts
from src.extract_keywords import KeywordHit, counts_to_ranked_hits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONCEPT_CONFIG_PATH = PROJECT_ROOT / "config" / "concept_clustering.yaml"

# Fallbacks used only if config/concept_clustering.yaml is missing or malformed.
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_MIN_CLUSTER_SIZE = 3
FALLBACK_MIN_SAMPLES = None

# HDBSCAN's noise label for points it declines to assign to any cluster.
NOISE_LABEL = -1


@lru_cache(maxsize=1)
def get_concept_config(path: str = str(DEFAULT_CONCEPT_CONFIG_PATH)) -> dict:
    """Load embedding/clustering settings from config/concept_clustering.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {
            "model_name": FALLBACK_MODEL_NAME,
            "min_cluster_size": FALLBACK_MIN_CLUSTER_SIZE,
            "min_samples": FALLBACK_MIN_SAMPLES,
        }

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "model_name": payload.get("model_name", FALLBACK_MODEL_NAME),
        "min_cluster_size": int(payload.get("min_cluster_size", FALLBACK_MIN_CLUSTER_SIZE)),
        "min_samples": payload.get("min_samples", FALLBACK_MIN_SAMPLES),
    }


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str):
    """Load (and cache) the sentence-transformers model. Slow on first call
    per process -- downloads the model on first-ever use, then loads from
    the local cache."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_terms(terms: list[str], *, model_name: str | None = None) -> np.ndarray:
    """Embed each term as a unit-normalized vector."""
    cfg = get_concept_config()
    model = get_embedding_model(model_name or cfg["model_name"])
    return model.encode(terms, normalize_embeddings=True, show_progress_bar=False)


def cluster_terms(embeddings: np.ndarray) -> np.ndarray:
    """HDBSCAN-cluster embeddings. Returns one cluster id per row (-1 = noise).

    Cosine similarity on unit-normalized vectors is equivalent to Euclidean
    distance up to a monotonic transform, so plain 'euclidean' is used here
    -- HDBSCAN doesn't support a 'cosine' metric directly.
    """
    import hdbscan

    cfg = get_concept_config()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


def label_concepts(
    concept_members: dict[str, list[str]], chat_counts: Counter[str]
) -> dict[str, str]:
    """Pick each concept's most chat-frequent member term as its display label."""
    return {
        concept_id: max(members, key=lambda term: (chat_counts.get(term, 0), term))
        for concept_id, members in concept_members.items()
    }


def cluster_keywords_into_concepts(
    chat_keywords: list[dict],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster every distinct keyword across the corpus into concepts.

    Returns (term_to_concept_label, concept_label_to_member_terms). Noise
    points (HDBSCAN couldn't group them with anything) each become their
    own singleton concept keyed by their own term, so no keyword's signal
    is silently dropped -- it just doesn't get merged with anything else.
    """
    chat_counts = compute_chat_counts(chat_keywords)
    terms = sorted(chat_counts)
    if not terms:
        return {}, {}

    embeddings = embed_terms(terms)
    cluster_ids = cluster_terms(embeddings)

    concept_members: dict[str, list[str]] = defaultdict(list)
    for term, cluster_id in zip(terms, cluster_ids):
        key = f"__singleton__{term}" if cluster_id == NOISE_LABEL else f"cluster_{cluster_id}"
        concept_members[key].append(term)

    concept_labels = label_concepts(concept_members, chat_counts)

    term_to_concept: dict[str, str] = {}
    concept_to_terms: dict[str, list[str]] = {}
    for concept_key, members in concept_members.items():
        label = concept_labels[concept_key]
        concept_to_terms[label] = members
        for term in members:
            term_to_concept[term] = label

    return term_to_concept, concept_to_terms


def build_chat_concepts(
    chat_keywords: list[dict], term_to_concept: dict[str, str]
) -> list[dict[str, object]]:
    """Re-key each chat's keyword hits by concept label.

    Output has the same {"chat_id", "keywords": [{"term", "count"}]} shape
    as chat_keywords.json, so build_cooccurrence_graph() and
    compute_height_scores() work on it completely unchanged -- they just
    see concept labels instead of raw keywords as "terms". Keywords that
    map to the same concept within one chat have their counts summed.
    """
    results: list[dict[str, object]] = []
    for entry in chat_keywords:
        chat_id = entry.get("chat_id")
        concept_counts: Counter[str] = Counter()
        for hit in entry.get("keywords", []):
            term = hit.get("term")
            if not isinstance(term, str):
                continue
            concept = term_to_concept.get(term, term)
            concept_counts[concept] += int(hit.get("count", 1))

        hits: list[KeywordHit] = counts_to_ranked_hits(concept_counts)
        results.append({"chat_id": chat_id, "keywords": hits})

    return results

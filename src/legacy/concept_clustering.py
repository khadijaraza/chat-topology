"""Group extracted keywords into semantic concepts via embedding clustering.

Concepts, not raw keywords, become the graph's nodes: two keywords end up
in the same concept because they mean similar things (per sentence
embeddings), independent of whether they ever co-occur in a chat. Concept-
to-concept *edges* still come from co-occurrence -- see
build_chat_concepts(), which re-keys each chat's keyword hits by concept so
the existing co-occurrence graph builder (src/legacy/build_graph.py) can run
on concepts unmodified.

Superseded: both the chat-keyed pathway above AND the segment-keyed pathway
below (cluster_segment_keywords_into_concepts(), originally Phase 3 of the
reconstruction) are legacy -- the current pipeline's mid tier
(src/conversation_concepts.py) clusters segment EMBEDDINGS directly instead
of segment KEYWORDS. Kept, not deleted -- both were validated working code
in their time (see CLAUDE.md's "Pipeline reconstruction" for the full
story). get_embedding_model() moved OUT of this file to
src/text_embedding.py, since src.segmentation and src.grounded_extraction
(current pipeline) both still need that cached model loader -- import it
from there, not from here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.extract_keywords import KeywordHit, counts_to_ranked_hits
from src.legacy.build_graph import compute_chat_counts
from src.text_embedding import get_embedding_model, mean_pool_normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONCEPT_CONFIG_PATH = PROJECT_ROOT / "config" / "legacy" / "concept_clustering.yaml"

# Fallbacks used only if config/legacy/concept_clustering.yaml is missing or malformed.
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"
FALLBACK_MIN_CLUSTER_SIZE = 3
FALLBACK_MIN_SAMPLES = None
FALLBACK_ENTITY_LABEL_MULTIPLIER = 1.3
FALLBACK_MIN_LABEL_SEGMENT_FREQUENCY = 2

# HDBSCAN's noise label for points it declines to assign to any cluster.
NOISE_LABEL = -1

# NER labels treated as "specific, domain-defining anchors" for
# label_concepts_by_specificity -- a proper noun/title/org/place is a much
# better concept label than a generic abstract noun that happens to sit
# near a cluster's geometric center (see that function's docstring).
ENTITY_LABELS_BOOSTED = {"WORK_OF_ART", "PERSON", "ORG", "GPE"}


@lru_cache(maxsize=1)
def get_concept_config(path: str = str(DEFAULT_CONCEPT_CONFIG_PATH)) -> dict:
    """Load embedding/clustering settings from config/legacy/concept_clustering.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {
            "model_name": FALLBACK_MODEL_NAME,
            "min_cluster_size": FALLBACK_MIN_CLUSTER_SIZE,
            "min_samples": FALLBACK_MIN_SAMPLES,
            "entity_label_multiplier": FALLBACK_ENTITY_LABEL_MULTIPLIER,
            "min_label_segment_frequency": FALLBACK_MIN_LABEL_SEGMENT_FREQUENCY,
        }

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "model_name": payload.get("model_name", FALLBACK_MODEL_NAME),
        "min_cluster_size": int(payload.get("min_cluster_size", FALLBACK_MIN_CLUSTER_SIZE)),
        "min_samples": payload.get("min_samples", FALLBACK_MIN_SAMPLES),
        "entity_label_multiplier": float(
            payload.get("entity_label_multiplier", FALLBACK_ENTITY_LABEL_MULTIPLIER)
        ),
        "min_label_segment_frequency": int(
            payload.get("min_label_segment_frequency", FALLBACK_MIN_LABEL_SEGMENT_FREQUENCY)
        ),
    }


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


def label_concepts_by_specificity(
    concept_members: dict[str, list[str]],
    embeddings_by_term: dict[str, np.ndarray],
    segment_frequencies: Counter[str],
    entity_labels: dict[str, str | None],
    total_segments: int,
    entity_label_multiplier: float,
    min_label_segment_frequency: int,
) -> dict[str, str]:
    """Pick each concept's label by maximizing
    cosine_similarity(term, cluster_centroid) * inverse_segment_frequency *
    entity_specificity_multiplier, instead of raw chat frequency.

    Chat-frequency labeling (label_concepts) tends to pick generic abstract
    terms -- "tragic foreshadowing", "discussion", "approach" -- since they
    sit in dense, generalized semantic space close to a cluster's
    geometric center, while specific anchors like "plato's symposium" sit
    further out despite being the more useful label. Two corrections:

    1. inverse_segment_frequency = log((total_segments + 1) / (segment
       count containing this term + 1)) -- a classic inverse-document-
       frequency term, using segments as "documents". This is a deliberate
       reinterpretation of the originally-proposed "inverse cluster
       frequency" (log(N_clusters / N_clusters_containing_similar_terms)):
       HDBSCAN produces a hard partition, so a term literally belongs to
       exactly one cluster, and "clusters containing similar terms" has no
       well-defined denominator without an expensive separate cross-cluster
       similarity pass. Segment frequency achieves the same goal (penalize
       terms that show up everywhere, reward domain-specific rarities)
       with a well-defined, already-available signal.
    2. entity_label_multiplier is applied to terms extracted as one of
       ENTITY_LABELS_BOOSTED (WORK_OF_ART/PERSON/ORG/GPE) -- specific named
       entities are close to always better labels than a generic noun
       phrase, even before the frequency-based scoring kicks in.

    Real-data correction: rewarding rarity rewards ALL rarity, not just
    genuine domain specificity -- on real data this picked "sam jose" (a
    literal typo of "san jose", segment_frequency=1) as a concept's label
    over the correctly-spelled, far more frequent "san jose", precisely
    because being rare inflated its specificity score. min_label_segment_
    frequency excludes any member below that segment count from being
    ELIGIBLE as the label (they can still be cluster members, just not the
    representative), falling back to considering every member if none
    clear the bar (a genuinely small/singleton concept still needs a label).
    """
    labels: dict[str, str] = {}
    for concept_id, members in concept_members.items():
        member_vectors = np.stack([embeddings_by_term[member] for member in members])
        centroid = mean_pool_normalize(member_vectors) if len(members) > 1 else member_vectors[0]

        eligible = [m for m in members if segment_frequencies.get(m, 0) >= min_label_segment_frequency]
        candidates = eligible or members

        best_term, best_score = candidates[0], -np.inf
        for member in candidates:
            centrality = float(embeddings_by_term[member] @ centroid)
            doc_freq = segment_frequencies.get(member, 1)
            specificity = np.log((total_segments + 1) / (doc_freq + 1))
            multiplier = entity_label_multiplier if entity_labels.get(member) in ENTITY_LABELS_BOOSTED else 1.0
            score = centrality * specificity * multiplier
            if score > best_score:
                best_term, best_score = member, score
        labels[concept_id] = best_term

    return labels


def _build_concepts_from_embeddings(
    terms: list[str],
    embeddings: np.ndarray,
    label_fn: Callable[[dict[str, list[str]]], dict[str, str]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Shared tail of both clustering pathways below: HDBSCAN the given
    embeddings, then label each resulting cluster via label_fn (either
    label_concepts or label_concepts_by_specificity, bound by the caller).
    Noise points each become their own singleton concept keyed by their
    own term, so no keyword's signal is silently dropped -- it just
    doesn't get merged with anything else."""
    cluster_ids = cluster_terms(embeddings)

    concept_members: dict[str, list[str]] = defaultdict(list)
    for term, cluster_id in zip(terms, cluster_ids):
        key = f"__singleton__{term}" if cluster_id == NOISE_LABEL else f"cluster_{cluster_id}"
        concept_members[key].append(term)

    concept_labels = label_fn(concept_members)

    term_to_concept: dict[str, str] = {}
    concept_to_terms: dict[str, list[str]] = {}
    for concept_key, members in concept_members.items():
        label = concept_labels[concept_key]
        concept_to_terms[label] = members
        for term in members:
            term_to_concept[term] = label

    return term_to_concept, concept_to_terms


def cluster_keywords_into_concepts(
    chat_keywords: list[dict],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster every distinct keyword across the corpus into concepts,
    embedding each bare term in isolation (the original, chat-keyed
    pathway -- see cluster_segment_keywords_into_concepts for the
    context-aware one built for Phase 2's segment-level output).

    Returns (term_to_concept_label, concept_label_to_member_terms).
    """
    chat_counts = compute_chat_counts(chat_keywords)
    terms = sorted(chat_counts)
    if not terms:
        return {}, {}

    embeddings = embed_terms(terms)
    return _build_concepts_from_embeddings(
        terms, embeddings, lambda members: label_concepts(members, chat_counts)
    )


def compute_distinct_chat_counts(segment_keywords: list[dict]) -> Counter[str]:
    """Distinct CHAT count per term from segment-keyed entries (not
    segment count) -- a term appearing in several segments of the same
    chat still counts once, consistent with how chat_count is used
    everywhere else in this pipeline (concept labeling, eventual graph
    node prominence)."""
    chats_by_term: dict[str, set[str]] = defaultdict(set)
    for entry in segment_keywords:
        chat_id = entry.get("chat_id")
        if not isinstance(chat_id, str):
            continue
        for hit in entry.get("keywords", []):
            term = hit.get("term")
            if isinstance(term, str):
                chats_by_term[term].add(chat_id)
    return Counter({term: len(chats) for term, chats in chats_by_term.items()})


def build_term_contexts(segment_keywords: list[dict], context_size: int = 4) -> dict[str, str]:
    """For each distinct term, build a lightweight "phrase + context"
    embedding input: the term plus its top co-occurring keywords from
    whichever segment it scored highest in (its single most representative
    occurrence). Disambiguates polysemous single words using real
    co-occurring context rather than embedding the bare word alone -- e.g.
    "tip" alongside "hand"/"finger"/"gesture" embeds differently than "tip"
    alongside "advice"/"suggestion"/"recommend". This is what closes the
    embedding-space polysemy gap noted in CLAUDE.md (switching to a
    stronger model fixed the specific "fingertip"/"pro tip" collision
    found on real data, but didn't structurally prevent the same class of
    error elsewhere -- context does)."""
    best_score_for_term: dict[str, float] = {}
    context_for_term: dict[str, list[str]] = {}

    for entry in segment_keywords:
        hits = entry.get("keywords", [])
        terms_in_segment = [hit["term"] for hit in hits if isinstance(hit.get("term"), str)]
        for hit in hits:
            term = hit.get("term")
            if not isinstance(term, str):
                continue
            score = float(hit.get("score", 0.0))
            if term not in best_score_for_term or score > best_score_for_term[term]:
                best_score_for_term[term] = score
                context_for_term[term] = [t for t in terms_in_segment if t != term][:context_size]

    return {term: " ".join([term] + context) for term, context in context_for_term.items()}


def compute_segment_frequencies(segment_keywords: list[dict]) -> Counter[str]:
    """Distinct SEGMENT count per term -- the "document frequency"
    denominator for label_concepts_by_specificity's inverse-frequency
    scoring (segments stand in for "documents")."""
    counts: Counter[str] = Counter()
    for entry in segment_keywords:
        seen = {hit["term"] for hit in entry.get("keywords", []) if isinstance(hit.get("term"), str)}
        for term in seen:
            counts[term] += 1
    return counts


def extract_entity_labels(segment_keywords: list[dict]) -> dict[str, str | None]:
    """term -> the NER label it was extracted under (see
    grounded_extraction.generate_candidates), or None. Takes whichever
    label is seen first for a term across all its occurrences -- a given
    surface form should get a consistent entity label in practice."""
    labels: dict[str, str | None] = {}
    for entry in segment_keywords:
        for hit in entry.get("keywords", []):
            term = hit.get("term")
            if isinstance(term, str) and term not in labels:
                labels[term] = hit.get("entity_label")
    return labels


def cluster_segment_keywords_into_concepts(
    segment_keywords: list[dict],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster segment-level grounded keywords (Phase 2's output) into
    concepts. Unlike cluster_keywords_into_concepts, embeds each term WITH
    lightweight co-occurrence context (see build_term_contexts) instead of
    the bare term alone, and labels each concept by specificity (see
    label_concepts_by_specificity) instead of raw chat frequency.

    Returns (term_to_concept_label, concept_label_to_member_terms).
    """
    chat_counts = compute_distinct_chat_counts(segment_keywords)
    terms = sorted(chat_counts)
    if not terms:
        return {}, {}

    context_by_term = build_term_contexts(segment_keywords)
    embedding_inputs = [context_by_term.get(term, term) for term in terms]
    embeddings = embed_terms(embedding_inputs)
    embeddings_by_term = dict(zip(terms, embeddings))

    segment_frequencies = compute_segment_frequencies(segment_keywords)
    entity_labels = extract_entity_labels(segment_keywords)
    total_segments = len({entry.get("segment_id") for entry in segment_keywords})
    concept_cfg = get_concept_config()

    label_fn = lambda members: label_concepts_by_specificity(  # noqa: E731
        members,
        embeddings_by_term,
        segment_frequencies,
        entity_labels,
        total_segments,
        concept_cfg["entity_label_multiplier"],
        concept_cfg["min_label_segment_frequency"],
    )
    return _build_concepts_from_embeddings(terms, embeddings, label_fn)


def build_segment_concepts(
    segment_keywords: list[dict], term_to_concept: dict[str, str]
) -> list[dict[str, object]]:
    """Re-key each segment's keyword hits by concept label -- segment-keyed
    counterpart to build_chat_concepts. Takes the MAX score among keywords
    that map to the same concept within one segment (not a sum: cosine
    similarity scores don't have a clean interpretation added together)."""
    results: list[dict[str, object]] = []
    for entry in segment_keywords:
        concept_scores: dict[str, float] = {}
        for hit in entry.get("keywords", []):
            term = hit.get("term")
            if not isinstance(term, str):
                continue
            concept = term_to_concept.get(term, term)
            score = float(hit.get("score", 0.0))
            if concept not in concept_scores or score > concept_scores[concept]:
                concept_scores[concept] = score

        hits = sorted(
            ({"term": concept, "score": score} for concept, score in concept_scores.items()),
            key=lambda hit: -hit["score"],
        )
        results.append(
            {
                "segment_id": entry.get("segment_id"),
                "chat_id": entry.get("chat_id"),
                "label": entry.get("label"),
                "keywords": hits,
            }
        )
    return results


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

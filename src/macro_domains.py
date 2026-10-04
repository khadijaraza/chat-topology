"""Group segments into macro domains via multi-resolution graph community
detection (mutual k-NN graph + Leiden/CPM), not a fixed-count global
partition.

Top tier of the proximity visualization's 3-layer hierarchy (segments ->
conversation concepts -> macro domains -- see CLAUDE.md). Operates directly
on the 1,287 SEGMENT embeddings, not on conversation-concept centroids --
replaces the original agglomerative/Ward approach entirely after real data
showed its two structural failure modes:

1. **Ward's fixed-k hits a genuine resolution limit.** Forcing a fixed
   `n_clusters` (8-15) onto centroids of vastly different underlying
   density -- 96% of conversation concepts are HDBSCAN singletons, i.e.
   isolated raw segment embeddings, not real averaged clusters -- meant
   genuinely distinct communities kept getting merged into one oversized
   domain regardless of linkage choice (confirmed: "am2320 sensor" mixed
   philosophy, astronomy, and car repair even under ward). CPM (Constant
   Potts Model) doesn't have this resolution limit: its quality function
   compares each candidate community's internal density against a FIXED
   resolution parameter gamma, not against the size of the rest of the
   graph, so communities of very different sizes can coexist without one
   swallowing the other.
2. **Forcing 100% coverage distorts real domains.** The old Stage-2
   nearest-centroid fallback assigned every singleton concept to SOME
   domain, even ones with no real affinity to any of them (a one-off
   "webpack plugin" question has no real domain -- it just got assigned to
   whichever domain's centroid happened to be least-far). Fixed by an
   explicit NOISE_LABEL bucket ("Background / Minor Orbit"): segments in a
   too-small community, or whose average similarity to their own
   community's other members is too low, are routed there instead of
   distorting a real domain's centroid.

Labeling also changed: a single medoid segment's top keyword is fragile (one
specific keyword sitting closest to a cluster's geometric center says
nothing about whether it represents the whole cluster -- confirmed bug:
"korean skincare rec" as a label for a cluster that also contained
"webpack plugin"). TF-ICF (term frequency, inverse CLUSTER frequency --
distinct from conversation_concepts.py's inverse SEGMENT frequency) scores
every term that appears anywhere in a community by how concentrated it is
in that community vs. spread across all communities, and the label is a
SINGLE tight anchor term (<= max_label_words words) chosen by that score --
not a chained list of several terms (an earlier version joined the top 3
with "&"; real output read as noisy/long rather than a clean name).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MACRO_DOMAINS_CONFIG_PATH = PROJECT_ROOT / "config" / "macro_domains.yaml"

# Fallbacks used only if config/macro_domains.yaml is missing or malformed.
FALLBACK_KNN_K = 15
FALLBACK_CPM_RESOLUTION = 0.05
FALLBACK_MIN_COMMUNITY_SIZE = 5
FALLBACK_MIN_AVG_SIMILARITY = 0.25
FALLBACK_MAX_LABEL_WORDS = 3

# Segments that don't belong to any real community -- either their own
# community was too small, or they personally sit too far from their
# community's other members. Not a domain: deliberately excluded from
# TF-ICF's cluster-frequency denominator and never given a "salient theme"
# label, since by construction it has none.
BACKGROUND_LABEL = "Background / Minor Orbit"


@lru_cache(maxsize=1)
def get_macro_domains_config(path: str = str(DEFAULT_MACRO_DOMAINS_CONFIG_PATH)) -> dict:
    """Load mutual-kNN/Leiden/TF-ICF settings from config/macro_domains.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return {
            "knn_k": FALLBACK_KNN_K,
            "cpm_resolution": FALLBACK_CPM_RESOLUTION,
            "min_community_size": FALLBACK_MIN_COMMUNITY_SIZE,
            "min_avg_similarity": FALLBACK_MIN_AVG_SIMILARITY,
            "max_label_words": FALLBACK_MAX_LABEL_WORDS,
        }

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {
        "knn_k": int(payload.get("knn_k", FALLBACK_KNN_K)),
        "cpm_resolution": float(payload.get("cpm_resolution", FALLBACK_CPM_RESOLUTION)),
        "min_community_size": int(payload.get("min_community_size", FALLBACK_MIN_COMMUNITY_SIZE)),
        "min_avg_similarity": float(payload.get("min_avg_similarity", FALLBACK_MIN_AVG_SIMILARITY)),
        "max_label_words": int(payload.get("max_label_words", FALLBACK_MAX_LABEL_WORDS)),
    }


def build_mutual_knn_graph(embeddings: np.ndarray, k: int):
    """A segment-to-segment graph with an edge only where two segments are
    MUTUALLY in each other's top-k nearest neighbors (cosine similarity on
    these unit-normalized embeddings == dot product).

    Mutual (not one-directional) kNN matters: a one-directional kNN graph
    lets a single popular/central segment become a hub connected to many
    segments that don't actually resemble each other, which is exactly the
    kind of false bridging that would recreate Ward's catch-all problem
    inside the graph itself. Requiring mutuality only keeps edges between
    segments that are each genuinely one of the other's closest matches.
    """
    import igraph as ig

    n = len(embeddings)
    if n == 0:
        return ig.Graph()

    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)

    k = min(k, n - 1) if n > 1 else 0
    if k <= 0:
        return ig.Graph(n=n)

    # Top-k neighbor indices per row (unordered within the top-k set --
    # exact rank among the k doesn't matter, only membership does).
    knn_idx = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]

    is_knn = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    is_knn[rows, knn_idx.ravel()] = True

    mutual = is_knn & is_knn.T
    edge_rows, edge_cols = np.nonzero(np.triu(mutual, k=1))
    edges = list(zip(edge_rows.tolist(), edge_cols.tolist()))

    return ig.Graph(n=n, edges=edges)


def run_leiden_cpm(graph, resolution: float, seed: int = 42) -> list[int]:
    """Leiden community detection under the Constant Potts Model.

    CPM's quality function evaluates each community against a FIXED
    resolution parameter (not against total graph size, unlike modularity),
    which is what avoids modularity/Ward-style resolution limits -- a
    genuinely tight, small community and a genuinely tight, large community
    can both be accepted on their own merits instead of the small one
    always losing to a merge. Returns one community id per node (isolated
    nodes each end up as their own singleton community, not -1 --
    filtered out downstream by min_community_size instead).
    """
    import leidenalg

    if graph.vcount() == 0:
        return []

    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )
    return list(partition.membership)


def _average_intra_community_similarity(
    embeddings: np.ndarray, membership: list[int]
) -> np.ndarray:
    """Per-segment: mean cosine similarity to every OTHER member of its own
    community. A segment nominally placed in a real-sized community by
    Leiden can still individually be a weak, borderline member -- this
    catches that case, which community-SIZE filtering alone wouldn't."""
    n = len(membership)
    result = np.zeros(n, dtype=np.float64)
    members_by_community: dict[int, list[int]] = defaultdict(list)
    for row, community in enumerate(membership):
        members_by_community[community].append(row)

    for rows in members_by_community.values():
        if len(rows) <= 1:
            continue
        sub = embeddings[rows]
        sims = sub @ sub.T
        np.fill_diagonal(sims, np.nan)
        result[rows] = np.nanmean(sims, axis=1)

    return result


def label_communities_by_tf_icf(
    domain_to_segments: dict[str, list[str]],
    segment_keywords: dict[str, list[str]],
    max_label_words: int,
) -> dict[str, str]:
    """Label each real community (BACKGROUND_LABEL excluded) with a SINGLE
    tight anchor term under TF-ICF: term frequency within the community,
    weighted by how UNEVENLY that term is spread across all communities.

    tf(term, community) = number of distinct segments in the community
    whose grounded keyword list contains the term (presence, not raw
    count, so one segment repeating a term can't dominate).
    icf(term) = log((n_communities + 1) / (communities containing term + 1))
    -- classic inverse-document-frequency smoothing, communities as
    "documents" instead of conversation_concepts.py's segments-as-documents.

    This directly targets the bug that motivated it: a single medoid
    segment's top keyword can be an outlier that doesn't represent the rest
    of the cluster at all. Multiplying by term SALIENCE across the whole
    community, not one segment's ranking, fixes that -- though the result
    is still whatever real extracted keyword phrase scores highest, not a
    hand-composed category name.

    A label is the single highest-scoring term that's already <=
    max_label_words words long -- not a chained "term & term & term" list
    (an earlier version of this function did that; real output read as
    noisy/long rather than as a clean anchor name). Falls back to
    truncating the single highest-scoring term's first max_label_words
    words only on the rare community where no real extracted term is
    already short enough on its own.
    """
    real_domains = {label: segs for label, segs in domain_to_segments.items() if label != BACKGROUND_LABEL}

    term_presence_by_domain: dict[str, Counter[str]] = {}
    for label, segment_ids in real_domains.items():
        presence: Counter[str] = Counter()
        for segment_id in segment_ids:
            for term in set(segment_keywords.get(segment_id, [])):
                presence[term] += 1
        term_presence_by_domain[label] = presence

    n_communities = len(real_domains)
    communities_containing_term: Counter[str] = Counter()
    for presence in term_presence_by_domain.values():
        for term in presence:
            communities_containing_term[term] += 1

    def icf(term: str) -> float:
        return math.log((n_communities + 1) / (communities_containing_term[term] + 1)) + 1.0

    labels: dict[str, str] = {}
    for label, presence in term_presence_by_domain.items():
        scored = sorted(
            presence.items(),
            key=lambda item: (-(item[1] * icf(item[0])), item[0]),
        )
        short_term = next((term for term, _count in scored if len(term.split()) <= max_label_words), None)
        if short_term is not None:
            labels[label] = short_term
        elif scored:
            labels[label] = " ".join(scored[0][0].split()[:max_label_words])
        else:
            labels[label] = "(untitled domain)"

    return labels


def cluster_into_macro_domains(
    segment_ids: list[str],
    embeddings: np.ndarray,
    segment_keywords: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster segments directly into macro domains via mutual-kNN + Leiden
    (CPM) + TF-ICF labeling + explicit noise routing.

    Returns (segment_id_to_domain_label, domain_label_to_member_segment_ids).
    Every segment appears in exactly one bucket, but that bucket may be
    BACKGROUND_LABEL -- unlike the old approach, coverage of "real" domains
    is NOT forced to be 100%.
    """
    if not segment_ids:
        return {}, {}

    cfg = get_macro_domains_config()
    graph = build_mutual_knn_graph(embeddings, cfg["knn_k"])
    membership = run_leiden_cpm(graph, cfg["cpm_resolution"])
    avg_similarity = _average_intra_community_similarity(embeddings, membership)

    # Two passes, not one: is_too_small must check how many members SURVIVE
    # the per-segment similarity filter, not Leiden's original community
    # size. Real bug found on real data: a community of size 6 could lose 5
    # members to the individual avg_similarity check and still pass the
    # size check (checked against the original 6), leaving a 1-segment
    # "real" domain -- exactly the noise-as-a-domain problem this bucket
    # exists to prevent, just arrived at a different way.
    is_too_dissimilar = avg_similarity < cfg["min_avg_similarity"]
    surviving_size: Counter[int] = Counter(
        community for row, community in enumerate(membership) if not is_too_dissimilar[row]
    )

    segment_to_domain: dict[str, str] = {}
    domain_to_segments_raw: dict[str, list[str]] = defaultdict(list)
    for row, segment_id in enumerate(segment_ids):
        community = membership[row]
        is_too_small = surviving_size[community] < cfg["min_community_size"]
        if is_too_small or is_too_dissimilar[row]:
            label = BACKGROUND_LABEL
        else:
            label = f"__community_{community}__"
        segment_to_domain[segment_id] = label
        domain_to_segments_raw[label].append(segment_id)

    community_labels = label_communities_by_tf_icf(
        domain_to_segments_raw, segment_keywords, cfg["max_label_words"]
    )

    domain_to_segments: dict[str, list[str]] = {}
    relabel: dict[str, str] = {}
    for raw_label, members in domain_to_segments_raw.items():
        final_label = raw_label if raw_label == BACKGROUND_LABEL else community_labels[raw_label]
        # Two different communities landing on the identical top-terms
        # label is possible (small vocab overlap) -- dedupe the same way
        # conversation_concepts.py does, so no community's members are
        # silently dropped by a dict-key collision.
        unique_label = final_label
        suffix = 2
        while unique_label in domain_to_segments:
            unique_label = f"{final_label} ({suffix})"
            suffix += 1
        domain_to_segments[unique_label] = members
        relabel[raw_label] = unique_label

    segment_to_domain = {seg: relabel[label] for seg, label in segment_to_domain.items()}

    return segment_to_domain, domain_to_segments


def assign_concepts_to_domains(
    concept_to_segments: dict[str, list[str]],
    segment_to_domain: dict[str, str],
) -> dict[str, str]:
    """Map each conversation concept (src/conversation_concepts.py's mid
    tier) to a macro domain by majority vote among its member segments'
    domain assignments.

    Domains are now computed directly from segments, not from conversation-
    concept centroids (see module docstring), so a concept -> domain
    mapping is no longer the primary clustering axis -- it's derived
    afterward, purely so the frontend's schema (every SegmentNode AND every
    ConversationConcept needs a macro_domain_id) has something to show.
    A multi-segment concept's segments CAN legitimately split across
    different domains (HDBSCAN's local segment-embedding neighborhoods and
    Leiden's mutual-kNN communities aren't the same partition) -- majority
    vote is a reasonable single answer for "where does this concept mostly
    live," not a claim that 100% of its segments agree.
    """
    result: dict[str, str] = {}
    for concept_label, segment_ids in concept_to_segments.items():
        votes = Counter(segment_to_domain.get(s, BACKGROUND_LABEL) for s in segment_ids)
        result[concept_label] = votes.most_common(1)[0][0]
    return result

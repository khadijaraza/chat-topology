"""Build a keyword co-occurrence graph with PMI-weighted edges.

Superseded: part of the original word-level pipeline (see CLAUDE.md's
"Pipeline reconstruction"). Proximity (UMAP on segment embeddings, see
src/proximity_layout.py) replaced graph edges entirely for the current
pipeline. Kept, not deleted -- validated working code.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GRAPH_CONFIG_PATH = PROJECT_ROOT / "config" / "legacy" / "graph.yaml"

# Fallbacks used only if config/legacy/graph.yaml is missing or malformed.
FALLBACK_MIN_CHAT_COUNT = 2
FALLBACK_MAX_EDGES_PER_NODE = None


@lru_cache(maxsize=1)
def get_graph_config(path: str = str(DEFAULT_GRAPH_CONFIG_PATH)) -> tuple[int, int | None]:
    """Load min_chat_count / max_edges_per_node from config/legacy/graph.yaml.

    Returns (min_chat_count, max_edges_per_node); max_edges_per_node is
    None when pruning is disabled (either by config or a missing file).
    """
    config_path = Path(path)
    if not config_path.is_file():
        return FALLBACK_MIN_CHAT_COUNT, FALLBACK_MAX_EDGES_PER_NODE

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    try:
        min_chat_count = int(payload.get("min_chat_count", FALLBACK_MIN_CHAT_COUNT))
    except (TypeError, ValueError):
        min_chat_count = FALLBACK_MIN_CHAT_COUNT

    raw_max_edges = payload.get("max_edges_per_node", FALLBACK_MAX_EDGES_PER_NODE)
    if raw_max_edges is None:
        max_edges_per_node = None
    else:
        try:
            max_edges_per_node = int(raw_max_edges)
        except (TypeError, ValueError):
            max_edges_per_node = FALLBACK_MAX_EDGES_PER_NODE

    return min_chat_count, max_edges_per_node


def terms_in_chat(entry: dict) -> set[str]:
    keywords = entry.get("keywords", [])
    terms: set[str] = set()
    if not isinstance(keywords, list):
        return terms
    for hit in keywords:
        if isinstance(hit, dict):
            term = hit.get("term")
            if isinstance(term, str) and term.strip():
                terms.add(term.strip())
    return terms


def compute_chat_counts(chat_keywords: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in chat_keywords:
        for term in terms_in_chat(entry):
            counts[term] += 1
    return counts


def pmi_from_counts(
    cooccurring_chats: int,
    chats_with_a: int,
    chats_with_b: int,
    total_chats: int,
) -> float:
    """Core PMI math shared by ``compute_pmi`` and graph construction."""
    if total_chats == 0 or cooccurring_chats == 0 or chats_with_a == 0 or chats_with_b == 0:
        return float("-inf")

    p_a = chats_with_a / total_chats
    p_b = chats_with_b / total_chats
    p_ab = cooccurring_chats / total_chats
    return math.log2(p_ab / (p_a * p_b))


def compute_pmi(term_a: str, term_b: str, chat_keywords: list[dict]) -> float:
    """Pointwise mutual information for binary chat-level term presence.

    PMI(A, B) = log2(P(A, B) / (P(A) * P(B)))

    Each chat contributes at most once to co-occurrence, matching the graph's
    intra-chat pairing logic. Returns ``float('-inf')`` when the terms never
    co-occur in the same chat.

    This is a standalone function (independent of ``build_cooccurrence_graph``)
    so you can inspect or tune the PMI signal for any two terms without
    rebuilding the whole graph.
    """
    if term_a == term_b:
        return float("inf")

    total_chats = len(chat_keywords)
    chats_with_a = 0
    chats_with_b = 0
    chats_with_both = 0

    for entry in chat_keywords:
        terms = terms_in_chat(entry)
        has_a = term_a in terms
        has_b = term_b in terms
        if has_a:
            chats_with_a += 1
        if has_b:
            chats_with_b += 1
        if has_a and has_b:
            chats_with_both += 1

    return pmi_from_counts(chats_with_both, chats_with_a, chats_with_b, total_chats)


def pmi_boost(pmi: float) -> float:
    """Non-negative PMI contribution used as an edge-weight boost.

    Negative PMI (terms co-occur less than chance) and undefined/degenerate
    values contribute nothing, so only positive statistical correlation adds
    extra weight on top of raw intra-chat co-occurrence counts.
    """
    if math.isinf(pmi) and pmi > 0:
        return 0.0
    if math.isnan(pmi) or math.isinf(pmi):
        return 0.0
    return max(0.0, pmi)


def prune_edges_to_top_k_per_node(graph: nx.Graph, max_edges_per_node: int) -> nx.Graph:
    """Keep an edge only if it's among one of its endpoints' strongest
    ``max_edges_per_node`` edges by weight.

    ``min_chat_count`` alone doesn't bound total edge count: a single
    keyword-rich chat contributes ``C(k, 2)`` edges for its ``k`` surviving
    keywords, so a handful of very long conversations can still make the
    graph near-complete regardless of how aggressively one-off keywords are
    filtered (on real data: 823 chats, min_chat_count=2, produced 10,316
    nodes but 7,886,476 edges -- one 2,892-keyword chat alone accounts for
    ~4.2M possible pairs). Keeping an edge if it's in *either* endpoint's
    top-K (rather than requiring both) avoids starving a node that has one
    dominant tie to a very well-connected partner. Bounds total edges to at
    most ``N * max_edges_per_node``.
    """
    keep: set[frozenset[str]] = set()
    for node in graph.nodes:
        strongest = sorted(
            graph.edges(node, data="weight"),
            key=lambda item: item[2],
            reverse=True,
        )
        for term_a, term_b, _weight in strongest[:max_edges_per_node]:
            keep.add(frozenset((term_a, term_b)))

    pruned = nx.Graph()
    pruned.add_nodes_from(graph.nodes(data=True))
    for term_a, term_b, data in graph.edges(data=True):
        if frozenset((term_a, term_b)) in keep:
            pruned.add_edge(term_a, term_b, **data)
    return pruned


def build_cooccurrence_graph(
    chat_keywords: list[dict],
    *,
    min_chat_count: int | None = None,
    max_edges_per_node: int | None = None,
) -> nx.Graph:
    """Build an undirected keyword graph from per-chat keyword lists.

    - One node per unique keyword, with a ``chat_count`` attribute (number of
      chats the keyword appears in).
    - One edge per pair of keywords that co-occur in at least one chat, with
      ``cooccurrence`` (raw intra-chat co-occurrence count), ``pmi``
      (pointwise mutual information across the corpus), ``pmi_boost`` (the
      non-negative part of PMI), and ``weight`` = cooccurrence + pmi_boost.
    - Nodes appearing in fewer than ``min_chat_count`` chats are dropped
      before edges are computed, so one-off noise keywords don't pollute the
      graph or count toward any other keyword's co-occurrence total.
    - If ``max_edges_per_node`` is set, edges are then pruned to each node's
      strongest ties (see prune_edges_to_top_k_per_node) -- necessary on top
      of min_chat_count, see that function's docstring for why.

    ``min_chat_count`` / ``max_edges_per_node`` default to config/legacy/graph.yaml
    when not given explicitly (pass a number to override for one-off tuning
    without touching the config file; max_edges_per_node=None disables
    pruning).
    """
    config_min_chat_count, config_max_edges_per_node = get_graph_config()
    if min_chat_count is None:
        min_chat_count = config_min_chat_count
    if max_edges_per_node is None:
        max_edges_per_node = config_max_edges_per_node

    total_chats = len(chat_keywords)
    chat_counts = compute_chat_counts(chat_keywords)
    kept_terms = {term for term, count in chat_counts.items() if count >= min_chat_count}

    graph = nx.Graph()
    for term in kept_terms:
        graph.add_node(term, chat_count=int(chat_counts[term]))

    cooccurrence: Counter[tuple[str, str]] = Counter()
    for entry in chat_keywords:
        terms = sorted(terms_in_chat(entry) & kept_terms)
        for term_a, term_b in combinations(terms, 2):
            cooccurrence[(term_a, term_b)] += 1

    for (term_a, term_b), intra_weight in cooccurrence.items():
        pmi = pmi_from_counts(
            intra_weight,
            chat_counts[term_a],
            chat_counts[term_b],
            total_chats,
        )
        boost = pmi_boost(pmi)
        graph.add_edge(
            term_a,
            term_b,
            cooccurrence=intra_weight,
            pmi=pmi,
            pmi_boost=boost,
            weight=intra_weight + boost,
        )

    if max_edges_per_node is not None:
        graph = prune_edges_to_top_k_per_node(graph, max_edges_per_node)

    return graph


def top_weighted_edges(graph: nx.Graph, limit: int = 10) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    for term_a, term_b, data in graph.edges(data=True):
        weight = float(data.get("weight", 0.0))
        edges.append((term_a, term_b, weight))
    edges.sort(key=lambda item: item[2], reverse=True)
    return edges[:limit]

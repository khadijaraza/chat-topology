"""Build a keyword co-occurrence graph with PMI-weighted edges."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Iterable

import networkx as nx


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
    """Non-negative PMI contribution used as an edge-weight boost."""
    if math.isinf(pmi) and pmi > 0:
        return 0.0
    if math.isnan(pmi) or math.isinf(pmi):
        return 0.0
    return max(0.0, pmi)


def build_cooccurrence_graph(
    chat_keywords: list[dict],
    *,
    min_chat_count: int = 2,
) -> nx.Graph:
    """Build an undirected keyword graph from per-chat keyword lists."""
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

    return graph


def top_weighted_edges(graph: nx.Graph, limit: int = 10) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    for term_a, term_b, data in graph.edges(data=True):
        weight = float(data.get("weight", 0.0))
        edges.append((term_a, term_b, weight))
    edges.sort(key=lambda item: item[2], reverse=True)
    return edges[:limit]

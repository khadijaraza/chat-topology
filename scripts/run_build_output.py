#!/usr/bin/env python3
"""Run the full pipeline end-to-end: ingest -> extraction -> concept clustering
-> graph -> layout -> height scores -> merge into data/processed/data.json.

This is the ONE command to run once every earlier stage has been validated
individually. It deliberately does NOT run phrase detection
(scripts/run_phrase_detection.py) or recompute concept clustering
(scripts/run_concept_clustering.py) -- both need a human to review their
output before it's baked into the rest of the pipeline (see CLAUDE.md), so
they stay separate, manually-triggered gates:

- If data/processed/phrases.json doesn't exist, extraction degrades
  gracefully to single-word keywords only.
- If data/processed/concepts.json doesn't exist, this run falls back to one
  concept per raw keyword (i.e. the old keyword-level graph). Otherwise it
  reapplies the reviewed keyword->concept mapping to this run's freshly
  extracted keywords -- cheap, and doesn't require re-embedding/re-
  clustering (which takes several minutes) on every run.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_graph import build_cooccurrence_graph, get_graph_config  # noqa: E402
from src.build_output import OUTPUT_PATH as DATA_JSON_PATH  # noqa: E402
from src.build_output import build_final_json, validate_schema  # noqa: E402
from src.concept_clustering import build_chat_concepts  # noqa: E402
from src.extract_keywords import extract_keywords  # noqa: E402
from src.height_score import compute_height_scores, get_height_config  # noqa: E402
from src.ingest import get_last_ingest_stats, load_all_conversations  # noqa: E402
from src.layout import compute_layout, get_coordinate_range  # noqa: E402

CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "chat_keywords.json"
CONCEPTS_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "concepts.json"
CHAT_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "chat_concepts.json"
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph.gpickle"
LAYOUT_PATH = PROJECT_ROOT / "data" / "processed" / "layout.json"
HEIGHT_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "height_scores.json"


def stage_ingest() -> list[dict]:
    conversations = load_all_conversations()
    stats = get_last_ingest_stats()
    print(
        f"[1/7] Ingest: {stats.total_loaded} conversations loaded, "
        f"{stats.total_skipped} skipped -> {CONVERSATIONS_PATH.relative_to(PROJECT_ROOT)}"
    )
    return conversations


def stage_extraction(conversations: list[dict]) -> list[dict]:
    results: list[dict] = []
    unique_keywords: set[str] = set()
    for conversation in tqdm(conversations, desc="[2/7] Extraction", unit="chat"):
        chat_id = conversation.get("chat_id")
        if not isinstance(chat_id, str):
            continue
        keywords = extract_keywords(conversation)
        results.append({"chat_id": chat_id, "keywords": keywords})
        unique_keywords.update(hit["term"] for hit in keywords)

    KEYWORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KEYWORDS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"      Extraction: {len(unique_keywords)} unique keywords "
        f"-> {KEYWORDS_PATH.relative_to(PROJECT_ROOT)}"
    )
    return results


def stage_concept_clustering(chat_keywords: list[dict]) -> list[dict]:
    """Apply concept clustering to this run's freshly extracted keywords.

    Reuses the reviewed mapping in data/processed/concepts.json (produced by
    a manual scripts/run_concept_clustering.py pass) rather than
    recomputing embeddings/clustering here -- see module docstring. Falls
    back to one concept per raw keyword if that review hasn't happened yet.
    """
    if CONCEPTS_SUMMARY_PATH.is_file():
        with CONCEPTS_SUMMARY_PATH.open(encoding="utf-8") as handle:
            concept_summary: dict[str, dict] = json.load(handle)
        term_to_concept = {
            term: label for label, info in concept_summary.items() for term in info["members"]
        }
        chat_concepts = build_chat_concepts(chat_keywords, term_to_concept)

        seen_terms = {hit["term"] for entry in chat_keywords for hit in entry.get("keywords", [])}
        unmapped = len(seen_terms - term_to_concept.keys())
        print(
            f"[3/7] Concept clustering: reused {CONCEPTS_SUMMARY_PATH.relative_to(PROJECT_ROOT)} "
            f"({len(concept_summary)} concepts); {unmapped} keywords from this run weren't in "
            f"that mapping and stayed ungrouped"
        )
    else:
        chat_concepts = chat_keywords
        print(
            f"[3/7] Concept clustering: {CONCEPTS_SUMMARY_PATH.relative_to(PROJECT_ROOT)} not "
            "found -- run scripts/run_concept_clustering.py and review its output first. "
            "Falling back to one concept per raw keyword for this run."
        )

    CHAT_CONCEPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_CONCEPTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chat_concepts, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return chat_concepts


def stage_graph_build(chat_concepts: list[dict]):
    min_chat_count, max_edges_per_node = get_graph_config()
    graph = build_cooccurrence_graph(chat_concepts)

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[4/7] Graph build: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges (min_chat_count={min_chat_count}, "
        f"max_edges_per_node={max_edges_per_node}, config/graph.yaml) "
        f"-> {GRAPH_PATH.relative_to(PROJECT_ROOT)}"
    )
    return graph


def stage_layout(graph) -> dict[str, tuple[float, float]]:
    layout = compute_layout(graph)
    coordinate_range = get_coordinate_range()

    payload = {term: [x, y] for term, (x, y) in layout.items()}
    LAYOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LAYOUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"[5/7] Layout: {len(layout)} nodes placed within "
        f"[-{coordinate_range:g}, {coordinate_range:g}] "
        f"-> {LAYOUT_PATH.relative_to(PROJECT_ROOT)} "
        "(preview PNG not regenerated here -- run scripts/run_layout.py for that)"
    )
    return layout


def stage_height_scores(graph, conversations: list[dict], chat_concepts: list[dict]) -> dict[str, float]:
    scores = compute_height_scores(graph, conversations, chat_concepts)
    _, z_range = get_height_config()

    HEIGHT_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HEIGHT_SCORES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"[6/7] Height scores: {len(scores)} topics scored within [0, {z_range:g}] "
        f"-> {HEIGHT_SCORES_PATH.relative_to(PROJECT_ROOT)}"
    )
    return scores


def stage_merge() -> dict:
    payload = build_final_json()

    DATA_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        f"[7/7] Merge: {len(payload['nodes'])} nodes, {len(payload['edges'])} edges, "
        f"{len(payload['supertopics'])} supertopics, {len(payload['superedges'])} superedges "
        f"-> {DATA_JSON_PATH.relative_to(PROJECT_ROOT)}"
    )
    return payload


def main() -> int:
    conversations = stage_ingest()
    if not conversations:
        print("ERROR: No conversations ingested -- check data/raw/<platform>/.", file=sys.stderr)
        return 1

    chat_keywords = stage_extraction(conversations)
    chat_concepts = stage_concept_clustering(chat_keywords)
    graph = stage_graph_build(chat_concepts)
    if graph.number_of_nodes() == 0:
        print("ERROR: Graph has no nodes -- nothing to lay out or score.", file=sys.stderr)
        return 1

    stage_layout(graph)
    stage_height_scores(graph, conversations, chat_concepts)
    payload = stage_merge()

    try:
        validate_schema(payload)
    except AssertionError as exc:
        print(f"\nSCHEMA VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print()
    print("Schema validation passed.")
    print()
    print("BEFORE YOU MOVE ON: data/processed/data.json is the frontend contract.")
    print("Copy it to frontend/public/data.json yourself when you want the app to")
    print("pick up this run -- that copy step stays manual on purpose.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

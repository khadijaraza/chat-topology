"""Merge graph, layout, and height-score artifacts into the frontend data contract.

This is the single Python -> frontend handoff: everything the Three.js scene
needs (node positions, sizes, colors-to-be, edges) in one file. See the
"nodes"/"edges" schema in build_final_json()'s docstring -- that shape is the
contract; changing field names here means updating the frontend reader too.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from src.build_graph import terms_in_chat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph.gpickle"
LAYOUT_PATH = PROJECT_ROOT / "data" / "processed" / "layout.json"
HEIGHT_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "height_scores.json"
CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
CHAT_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "chat_concepts.json"
SUPERTOPICS_PATH = PROJECT_ROOT / "data" / "processed" / "supertopics.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data.json"

SCHEMA_VERSION = 2

_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Turn a keyword into a URL/JS-safe id: lowercase, hyphen-separated."""
    slug = _SLUG_INVALID_RE.sub("-", text.strip().lower()).strip("-")
    return slug or "topic"


def _dedupe_slugs(labels: list[str]) -> dict[str, str]:
    """Slugify labels, suffixing -2, -3, ... on collision so ids stay unique.

    Two distinct keywords can slugify to the same string (e.g. "C++" and
    "c  " both -> "c"), and "id" must be a unique key per the frontend schema.
    A single global ``seen`` set (rather than a per-base counter) is required:
    with a per-base counter, a colliding label's disambiguated suffix (e.g.
    "t" and "t " both -> base "t", second one suffixed to "t-2") can still
    collide with a third label that slugifies to that exact suffix on its own
    (e.g. a literal keyword "T-2" also slugifying to "t-2") -- this happened
    on real data (10k+ keywords) and produced a real duplicate id.
    """
    seen: set[str] = set()
    slug_by_label: dict[str, str] = {}
    for label in labels:
        base = slugify(label)
        slug = base
        suffix = 2
        while slug in seen:
            slug = f"{base}-{suffix}"
            suffix += 1
        seen.add(slug)
        slug_by_label[label] = slug
    return slug_by_label


def _platforms_by_term(
    chat_keywords: list[dict], chat_platforms: dict[str, str]
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for entry in chat_keywords:
        chat_id = entry.get("chat_id")
        platform = chat_platforms.get(chat_id) if isinstance(chat_id, str) else None
        if not platform:
            continue
        for term in terms_in_chat(entry):
            result.setdefault(term, set()).add(platform)
    return result


def _load_concept_to_supertopic(supertopics_path: Path) -> dict[str, str]:
    """Invert supertopics.json (supertopic label -> member concepts) into
    concept -> supertopic label. Returns {} if the file doesn't exist yet
    (supertopic clustering is a manual review gate, like concept clustering
    and phrase detection -- see scripts/run_supertopic_clustering.py)."""
    if not supertopics_path.is_file():
        return {}
    with supertopics_path.open(encoding="utf-8") as handle:
        supertopics: dict[str, dict] = json.load(handle)
    return {
        concept: supertopic_label
        for supertopic_label, info in supertopics.items()
        for concept in info["members"]
    }


def _supertopic_chat_counts(
    chat_concepts: list[dict], concept_to_supertopic: dict[str, str]
) -> Counter[str]:
    """Distinct chat count per super-topic -- a chat touching several
    concepts in the same super-topic still only counts once, same as how
    graph node chat_count is computed at the concept level."""
    counts: Counter[str] = Counter()
    for entry in chat_concepts:
        touched = {concept_to_supertopic.get(term, term) for term in terms_in_chat(entry)}
        for supertopic in touched:
            counts[supertopic] += 1
    return counts


def build_final_json(
    *,
    graph_path: Path = GRAPH_PATH,
    layout_path: Path = LAYOUT_PATH,
    height_scores_path: Path = HEIGHT_SCORES_PATH,
    conversations_path: Path = CONVERSATIONS_PATH,
    chat_concepts_path: Path = CHAT_CONCEPTS_PATH,
    supertopics_path: Path = SUPERTOPICS_PATH,
) -> dict:
    """Merge graph.gpickle + layout.json + height_scores.json (+ supertopics.json,
    if it exists) into one payload:

    {
      "version": 2,
      "generated_at": "<ISO 8601 UTC timestamp>",
      "nodes": [
        {"id": str, "label": str, "x": float, "y": float, "z": float,
         "chat_count": int, "platforms": [str, ...], "supertopic": str | null},
        ...
      ],
      "edges": [{"source": str, "target": str, "weight": float}, ...],
      "supertopics": [
        {"id": str, "label": str, "x": float, "y": float, "z": float,
         "chat_count": int, "concept_count": int, "platforms": [str, ...]},
        ...
      ],
      "superedges": [{"source": str, "target": str, "weight": float}, ...]
    }

    "id" is the slugified concept label (disambiguated on collision, see
    _dedupe_slugs); "label" is the concept's representative keyword text
    (see src/concept_clustering.py -- a node is a semantic cluster of
    keywords, not a single raw keyword). "platforms" is derived from which
    chats mention the concept -- currently always ["gemini"] until
    Claude/ChatGPT ingestion exists (see src/ingest.py), but this stays
    correct without changes once that's added.

    "supertopics" is the second, coarser clustering layer (see
    src/supertopic_clustering.py) for the frontend's collapsed/zoomed-out
    view. A super-topic's (x, y, z) is the centroid of its member concept
    nodes' positions -- not an independently laid-out position -- and its
    "chat_count" is a true distinct-chat count (a chat touching several
    concepts in the same super-topic only counts once), not a sum. "edges"
    between concepts in the SAME super-topic are dropped when rolling up
    into "superedges"; only cross-super-topic weight survives, summed.
    Every node's "supertopic" is null, and "supertopics"/"superedges" are
    empty, if supertopics.json doesn't exist yet (manual review gate, like
    concept clustering -- run scripts/run_supertopic_clustering.py first).
    """
    with graph_path.open("rb") as handle:
        graph: nx.Graph = pickle.load(handle)
    with layout_path.open(encoding="utf-8") as handle:
        layout: dict[str, list[float]] = json.load(handle)
    with height_scores_path.open(encoding="utf-8") as handle:
        height_scores: dict[str, float] = json.load(handle)
    with conversations_path.open(encoding="utf-8") as handle:
        conversations: list[dict] = json.load(handle)
    with chat_concepts_path.open(encoding="utf-8") as handle:
        chat_concepts: list[dict] = json.load(handle)

    chat_platforms = {
        conv["chat_id"]: conv.get("platform", "unknown")
        for conv in conversations
        if isinstance(conv.get("chat_id"), str)
    }
    platforms_by_term = _platforms_by_term(chat_concepts, chat_platforms)

    terms = sorted(graph.nodes)
    slug_by_label = _dedupe_slugs(terms)

    concept_to_supertopic = _load_concept_to_supertopic(supertopics_path)
    supertopic_slug_by_label = _dedupe_slugs(sorted(set(concept_to_supertopic.values())))

    nodes = []
    positions: dict[str, tuple[float, float, float]] = {}
    for term in terms:
        x, y = layout.get(term, [0.0, 0.0])
        z = float(height_scores.get(term, 0.0))
        positions[term] = (float(x), float(y), z)
        supertopic_label = concept_to_supertopic.get(term)
        nodes.append(
            {
                "id": slug_by_label[term],
                "label": term,
                "x": float(x),
                "y": float(y),
                "z": z,
                "chat_count": int(graph.nodes[term].get("chat_count", 0)),
                "platforms": sorted(platforms_by_term.get(term, set())),
                "supertopic": supertopic_slug_by_label.get(supertopic_label) if supertopic_label else None,
            }
        )

    edges = [
        {
            "source": slug_by_label[term_a],
            "target": slug_by_label[term_b],
            "weight": float(data.get("weight", 0.0)),
        }
        for term_a, term_b, data in graph.edges(data=True)
    ]

    supertopics: list[dict] = []
    superedges: list[dict] = []

    if concept_to_supertopic:
        members_by_supertopic: dict[str, list[str]] = defaultdict(list)
        for term in terms:
            supertopic_label = concept_to_supertopic.get(term)
            if supertopic_label:
                members_by_supertopic[supertopic_label].append(term)

        supertopic_chat_counts = _supertopic_chat_counts(chat_concepts, concept_to_supertopic)

        for supertopic_label, members in members_by_supertopic.items():
            xs = [positions[m][0] for m in members]
            ys = [positions[m][1] for m in members]
            zs = [positions[m][2] for m in members]
            platforms: set[str] = set()
            for member in members:
                platforms |= platforms_by_term.get(member, set())
            supertopics.append(
                {
                    "id": supertopic_slug_by_label[supertopic_label],
                    "label": supertopic_label,
                    "x": sum(xs) / len(xs),
                    "y": sum(ys) / len(ys),
                    "z": sum(zs) / len(zs),
                    "chat_count": supertopic_chat_counts.get(supertopic_label, 0),
                    "concept_count": len(members),
                    "platforms": sorted(platforms),
                }
            )

        superedge_weights: Counter[tuple[str, str]] = Counter()
        for term_a, term_b, data in graph.edges(data=True):
            supertopic_a = concept_to_supertopic.get(term_a)
            supertopic_b = concept_to_supertopic.get(term_b)
            if not supertopic_a or not supertopic_b or supertopic_a == supertopic_b:
                continue
            key = tuple(sorted((supertopic_a, supertopic_b)))
            superedge_weights[key] += float(data.get("weight", 0.0))

        superedges = [
            {
                "source": supertopic_slug_by_label[a],
                "target": supertopic_slug_by_label[b],
                "weight": weight,
            }
            for (a, b), weight in superedge_weights.items()
        ]

    return {
        "version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "nodes": nodes,
        "edges": edges,
        "supertopics": supertopics,
        "superedges": superedges,
    }


def validate_schema(payload: dict) -> None:
    """Assert ``payload`` matches the frontend data contract; raises AssertionError."""
    assert payload.get("version") == SCHEMA_VERSION, f"version must be {SCHEMA_VERSION}"
    assert isinstance(payload.get("generated_at"), str) and payload["generated_at"], (
        "generated_at must be a non-empty ISO 8601 string"
    )
    assert isinstance(payload.get("nodes"), list), "nodes must be a list"
    assert isinstance(payload.get("edges"), list), "edges must be a list"
    assert isinstance(payload.get("supertopics"), list), "supertopics must be a list"
    assert isinstance(payload.get("superedges"), list), "superedges must be a list"

    supertopic_ids: set[str] = set()
    for supertopic in payload["supertopics"]:
        supertopic_id = supertopic.get("id")
        assert isinstance(supertopic_id, str) and supertopic_id, "supertopic.id must be a non-empty string"
        assert supertopic_id not in supertopic_ids, f"duplicate supertopic id: {supertopic_id!r}"
        supertopic_ids.add(supertopic_id)
        assert isinstance(supertopic.get("label"), str) and supertopic["label"], "supertopic.label required"
        for key in ("x", "y", "z"):
            assert isinstance(supertopic.get(key), (int, float)), f"supertopic.{key} must be numeric"
        assert isinstance(supertopic.get("chat_count"), int), "supertopic.chat_count must be an int"
        assert isinstance(supertopic.get("concept_count"), int), "supertopic.concept_count must be an int"
        assert isinstance(supertopic.get("platforms"), list) and all(
            isinstance(p, str) for p in supertopic["platforms"]
        ), "supertopic.platforms must be a list of strings"

    for edge in payload["superedges"]:
        assert edge.get("source") in supertopic_ids, f"superedge source not in supertopics: {edge.get('source')!r}"
        assert edge.get("target") in supertopic_ids, f"superedge target not in supertopics: {edge.get('target')!r}"
        assert isinstance(edge.get("weight"), (int, float)), "superedge.weight must be numeric"

    seen_ids: set[str] = set()
    for node in payload["nodes"]:
        node_id = node.get("id")
        assert isinstance(node_id, str) and node_id, "node.id must be a non-empty string"
        assert node_id not in seen_ids, f"duplicate node id: {node_id!r}"
        seen_ids.add(node_id)
        assert isinstance(node.get("label"), str) and node["label"], "node.label required"
        for key in ("x", "y", "z"):
            assert isinstance(node.get(key), (int, float)), f"node.{key} must be numeric"
        assert isinstance(node.get("chat_count"), int), "node.chat_count must be an int"
        assert isinstance(node.get("platforms"), list) and all(
            isinstance(p, str) for p in node["platforms"]
        ), "node.platforms must be a list of strings"
        supertopic = node.get("supertopic")
        assert supertopic is None or supertopic in supertopic_ids, (
            f"node.supertopic references unknown supertopic id: {supertopic!r}"
        )
        if supertopic_ids:
            assert supertopic is not None, f"node {node_id!r} has no supertopic but supertopics exist"

    for edge in payload["edges"]:
        assert edge.get("source") in seen_ids, f"edge source not in nodes: {edge.get('source')!r}"
        assert edge.get("target") in seen_ids, f"edge target not in nodes: {edge.get('target')!r}"
        assert isinstance(edge.get("weight"), (int, float)), "edge.weight must be numeric"

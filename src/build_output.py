"""Merge graph, layout, and height-score artifacts into the frontend data contract.

This is the single Python -> frontend handoff: everything the Three.js scene
needs. Mixed file, stays in src/ root: build_final_json()/validate_schema()
(schema v2, node/edge graph) are legacy-only, called only by
scripts/legacy/run_build_output.py; build_final_json_v3()/
validate_schema_v3() (schema v3, segments/concepts/domains, no edges) are
the current pipeline's output, called by scripts/run_build_output_v3.py.
See each function's own docstring for its schema.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from src.legacy.build_graph import terms_in_chat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "graph.gpickle"
LAYOUT_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "layout.json"
HEIGHT_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "height_scores.json"
CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "conversations.json"
CHAT_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "chat_concepts.json"
SUPERTOPICS_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "supertopics.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data.json"
# build_final_json() (v2) gets its OWN output path, not OUTPUT_PATH above --
# real latent bug, caught while sorting data/processed/ into current/legacy:
# both build_final_json() and build_final_json_v3() previously wrote to the
# exact same OUTPUT_PATH, so running the legacy pipeline's
# scripts/legacy/run_build_output.py after the current pipeline would
# silently overwrite the canonical schema-v3 data.json the frontend actually
# reads, with no warning. The two pipelines' outputs now live in genuinely
# separate files.
LEGACY_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "data.json"

SCHEMA_VERSION = 2

# Schema v3: the proximity-based 3-tier pipeline (segments -> conversation
# concepts -> macro domains, see CLAUDE.md's "Pipeline reconstruction").
# Supersedes v2 as what the frontend actually reads once wired in, but v2's
# build_final_json() stays intact and unused rather than deleted -- same
# "superseded, not deleted" treatment already given to the word-level
# src/legacy/concept_clustering.py and src/legacy/supertopic_clustering.py
# pathway it depends on.
SCHEMA_VERSION_V3 = 3

SEGMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "segments.json"
SEGMENT_KEYWORDS_PATH = PROJECT_ROOT / "data" / "processed" / "segment_keywords.json"
CONVERSATION_CONCEPTS_PATH = PROJECT_ROOT / "data" / "processed" / "conversation_concepts.json"
MACRO_DOMAINS_PATH = PROJECT_ROOT / "data" / "processed" / "macro_domains.json"
PROXIMITY_LAYOUT_PATH = PROJECT_ROOT / "data" / "processed" / "proximity_layout.json"
DENSITY_FIELD_PATH = PROJECT_ROOT / "data" / "processed" / "density_field.json"

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
    and phrase detection -- see scripts/legacy/run_supertopic_clustering.py)."""
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
    (see src/legacy/concept_clustering.py -- a node is a semantic cluster of
    keywords, not a single raw keyword). "platforms" is derived from which
    chats mention the concept -- currently always ["gemini"] until
    Claude/ChatGPT ingestion exists (see src/ingest.py), but this stays
    correct without changes once that's added.

    "supertopics" is the second, coarser clustering layer (see
    src/legacy/supertopic_clustering.py) for the frontend's collapsed/zoomed-out
    view. A super-topic's (x, y, z) is the centroid of its member concept
    nodes' positions -- not an independently laid-out position -- and its
    "chat_count" is a true distinct-chat count (a chat touching several
    concepts in the same super-topic only counts once), not a sum. "edges"
    between concepts in the SAME super-topic are dropped when rolling up
    into "superedges"; only cross-super-topic weight survives, summed.
    Every node's "supertopic" is null, and "supertopics"/"superedges" are
    empty, if supertopics.json doesn't exist yet (manual review gate, like
    concept clustering -- run scripts/legacy/run_supertopic_clustering.py first).
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


def _segment_labels(segment_keywords: list[dict]) -> dict[str, str]:
    """Each segment's top grounded keyword (highest MMR score) as its label.

    Not assumed to be keywords[0] -- MMR diversifies, it doesn't guarantee
    strict score order in the final list -- so this picks explicitly by max
    score rather than trusting position.
    """
    labels: dict[str, str] = {}
    for entry in segment_keywords:
        segment_id = entry.get("segment_id")
        keywords = entry.get("keywords") or []
        if not isinstance(segment_id, str) or not keywords:
            continue
        best = max(keywords, key=lambda kw: kw.get("score", 0.0))
        labels[segment_id] = best["term"]
    return labels


def build_final_json_v3(
    *,
    conversations_path: Path = CONVERSATIONS_PATH,
    segments_path: Path = SEGMENTS_PATH,
    segment_keywords_path: Path = SEGMENT_KEYWORDS_PATH,
    conversation_concepts_path: Path = CONVERSATION_CONCEPTS_PATH,
    macro_domains_path: Path = MACRO_DOMAINS_PATH,
    proximity_layout_path: Path = PROXIMITY_LAYOUT_PATH,
    density_field_path: Path = DENSITY_FIELD_PATH,
) -> dict:
    """Merge segments.json + segment_keywords.json + conversation_concepts.json
    + macro_domains.json + proximity_layout.json (+ density_field.json, if it
    exists) into the schema v3 frontend payload:

    {
      "version": 3,
      "generated_at": "<ISO 8601 UTC timestamp>",
      "coordinate_range": float,
      "density_field": {"x": [...], "y": [...], "normalized_density": [[...]],
                         "terrain_z_range": float} | null,
      "segments": [
        {"id": str, "label": str, "chat_id": str, "platform": str,
         "x": float, "y": float, "z": float, "word_count": int,
         "is_tangent": bool, "conversation_concept_id": str | null,
         "macro_domain_id": str | null},
        ...
      ],
      "conversation_concepts": [
        {"id": str, "label": str, "x": float, "y": float, "z": float,
         "segment_count": int, "macro_domain_id": str | null},
        ...
      ],
      "macro_domains": [
        {"id": str, "label": str, "x": float, "y": float, "z": float,
         "concept_count": int, "segment_count": int},
        ...
      ]
    }

    No "edges"/"superedges" -- proximity (x/y position) replaces the PMI
    co-occurrence graph entirely for this pathway, confirmed via
    AskUserQuestion (see CLAUDE.md's "Pipeline reconstruction"). A
    segment's "id" is its own segment_id (already a stable, unique,
    JS-safe string, "<chat_id>::seg<n>") -- not slugified, unlike
    concept/domain labels which are natural-language and need it.
    "label" is that segment's top grounded keyword (src/grounded_extraction.py,
    highest MMR score), not the raw "core"/"tangent" classification --
    "is_tangent" carries that instead. A concept/domain's (x, y, z) is the
    centroid of its member segments'/concepts' positions (src/proximity_layout.py's
    compute_group_centroids()), same pattern as v2's supertopic centroids.
    """
    with conversations_path.open(encoding="utf-8") as handle:
        conversations: list[dict] = json.load(handle)
    with segments_path.open(encoding="utf-8") as handle:
        segments_by_chat: dict[str, list[dict]] = json.load(handle)
    with segment_keywords_path.open(encoding="utf-8") as handle:
        segment_keywords: list[dict] = json.load(handle)
    with conversation_concepts_path.open(encoding="utf-8") as handle:
        concept_summary: dict[str, dict] = json.load(handle)
    with macro_domains_path.open(encoding="utf-8") as handle:
        domain_summary: dict[str, dict] = json.load(handle)
    with proximity_layout_path.open(encoding="utf-8") as handle:
        layout: dict = json.load(handle)

    # Optional, graceful degradation -- same pattern as v2's supertopics.json:
    # run_density_field.py is its own manual review gate
    # (scripts/run_density_field.py), not required for the rest of schema
    # v3 to work.
    density_field_out = None
    if density_field_path.is_file():
        with density_field_path.open(encoding="utf-8") as handle:
            density_field_raw: dict = json.load(handle)
        density_field_out = {
            "x": density_field_raw["x"],
            "y": density_field_raw["y"],
            "normalized_density": density_field_raw["normalized_density"],
            "terrain_z_range": density_field_raw["terrain_z_range"],
        }

    chat_platforms = {
        conv["chat_id"]: conv.get("platform", "unknown")
        for conv in conversations
        if isinstance(conv.get("chat_id"), str)
    }
    segment_label_by_id = _segment_labels(segment_keywords)

    concept_to_segments = {label: info["segment_ids"] for label, info in concept_summary.items()}
    # macro_domains.json's "segment_ids" is authoritative -- domains are
    # clustered directly from segments (src/macro_domains.py's mutual-kNN +
    # Leiden/CPM), not derived from concept membership. "concepts" there is
    # a separate, majority-vote-derived field (see
    # assign_concepts_to_domains()) used only for the ConversationConcept's
    # own macro_domain_id -- a concept's member segments CAN legitimately
    # span more than one domain, so segments must read their domain
    # directly, not through their parent concept.
    domain_to_segments = {label: info["segment_ids"] for label, info in domain_summary.items()}
    domain_to_concepts = {label: info["concepts"] for label, info in domain_summary.items()}
    segment_to_concept = {s: label for label, members in concept_to_segments.items() for s in members}
    segment_to_domain = {s: label for label, members in domain_to_segments.items() for s in members}
    concept_to_domain = {
        concept_label: domain_label
        for domain_label, members in domain_to_concepts.items()
        for concept_label in members
    }

    concept_slug_by_label = _dedupe_slugs(sorted(concept_to_segments))
    domain_slug_by_label = _dedupe_slugs(sorted(domain_to_segments))

    layout_segments: dict[str, dict] = layout.get("segments", {})
    layout_concepts: dict[str, dict] = layout.get("conversation_concepts", {})
    layout_domains: dict[str, dict] = layout.get("macro_domains", {})

    segments_out = []
    for chat_id, segs in segments_by_chat.items():
        platform = chat_platforms.get(chat_id, "unknown")
        for seg in segs:
            segment_id = seg["segment_id"]
            pos = layout_segments.get(segment_id)
            if pos is None:
                continue
            concept_label = segment_to_concept.get(segment_id)
            domain_label = segment_to_domain.get(segment_id)
            segments_out.append(
                {
                    "id": segment_id,
                    "label": segment_label_by_id.get(segment_id, ""),
                    "chat_id": chat_id,
                    "platform": platform,
                    "x": float(pos["x"]),
                    "y": float(pos["y"]),
                    "z": float(pos["z"]),
                    "word_count": int(seg.get("word_count", 0)),
                    "is_tangent": seg.get("label") == "tangent",
                    "conversation_concept_id": concept_slug_by_label.get(concept_label) if concept_label else None,
                    "macro_domain_id": domain_slug_by_label.get(domain_label) if domain_label else None,
                }
            )

    concepts_out = []
    for label, segment_ids in concept_to_segments.items():
        pos = layout_concepts.get(label)
        if pos is None:
            continue
        member_zs = [layout_segments[s]["z"] for s in segment_ids if s in layout_segments]
        domain_label = concept_to_domain.get(label)
        concepts_out.append(
            {
                "id": concept_slug_by_label[label],
                "label": label,
                "x": float(pos["x"]),
                "y": float(pos["y"]),
                "z": float(sum(member_zs) / len(member_zs)) if member_zs else 0.0,
                "segment_count": len(segment_ids),
                "macro_domain_id": domain_slug_by_label.get(domain_label) if domain_label else None,
            }
        )

    domains_out = []
    for label, member_segment_ids in domain_to_segments.items():
        pos = layout_domains.get(label)
        if pos is None:
            continue
        member_zs = [layout_segments[s]["z"] for s in member_segment_ids if s in layout_segments]
        domains_out.append(
            {
                "id": domain_slug_by_label[label],
                "label": label,
                "x": float(pos["x"]),
                "y": float(pos["y"]),
                "z": float(sum(member_zs) / len(member_zs)) if member_zs else 0.0,
                "concept_count": len(domain_to_concepts.get(label, [])),
                "segment_count": len(member_segment_ids),
            }
        )

    return {
        "version": SCHEMA_VERSION_V3,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "coordinate_range": float(layout.get("coordinate_range", 100.0)),
        "density_field": density_field_out,
        "segments": segments_out,
        "conversation_concepts": concepts_out,
        "macro_domains": domains_out,
    }


def validate_schema_v3(payload: dict) -> None:
    """Assert ``payload`` matches the schema v3 frontend data contract; raises AssertionError."""
    assert payload.get("version") == SCHEMA_VERSION_V3, f"version must be {SCHEMA_VERSION_V3}"
    assert isinstance(payload.get("generated_at"), str) and payload["generated_at"], (
        "generated_at must be a non-empty ISO 8601 string"
    )
    assert isinstance(payload.get("coordinate_range"), (int, float)), "coordinate_range must be numeric"
    assert isinstance(payload.get("segments"), list), "segments must be a list"
    assert isinstance(payload.get("conversation_concepts"), list), "conversation_concepts must be a list"
    assert isinstance(payload.get("macro_domains"), list), "macro_domains must be a list"

    density_field = payload.get("density_field")
    if density_field is not None:
        assert isinstance(density_field, dict), "density_field must be an object or null"
        for key in ("x", "y", "normalized_density"):
            assert isinstance(density_field.get(key), list), f"density_field.{key} must be a list"
        assert isinstance(density_field.get("terrain_z_range"), (int, float)), (
            "density_field.terrain_z_range must be numeric"
        )
        grid_size = len(density_field["x"])
        assert len(density_field["y"]) == grid_size, "density_field.x/y must be the same length"
        assert len(density_field["normalized_density"]) == grid_size, (
            "density_field.normalized_density must have grid_size rows"
        )

    domain_ids: set[str] = set()
    for domain in payload["macro_domains"]:
        domain_id = domain.get("id")
        assert isinstance(domain_id, str) and domain_id, "macro_domain.id must be a non-empty string"
        assert domain_id not in domain_ids, f"duplicate macro_domain id: {domain_id!r}"
        domain_ids.add(domain_id)
        assert isinstance(domain.get("label"), str) and domain["label"], "macro_domain.label required"
        for key in ("x", "y", "z"):
            assert isinstance(domain.get(key), (int, float)), f"macro_domain.{key} must be numeric"
        assert isinstance(domain.get("concept_count"), int), "macro_domain.concept_count must be an int"
        assert isinstance(domain.get("segment_count"), int), "macro_domain.segment_count must be an int"

    concept_ids: set[str] = set()
    for concept in payload["conversation_concepts"]:
        concept_id = concept.get("id")
        assert isinstance(concept_id, str) and concept_id, "conversation_concept.id must be a non-empty string"
        assert concept_id not in concept_ids, f"duplicate conversation_concept id: {concept_id!r}"
        concept_ids.add(concept_id)
        assert isinstance(concept.get("label"), str) and concept["label"], "conversation_concept.label required"
        for key in ("x", "y", "z"):
            assert isinstance(concept.get(key), (int, float)), f"conversation_concept.{key} must be numeric"
        assert isinstance(concept.get("segment_count"), int), "conversation_concept.segment_count must be an int"
        domain_id = concept.get("macro_domain_id")
        assert domain_id is None or domain_id in domain_ids, (
            f"conversation_concept.macro_domain_id references unknown macro_domain: {domain_id!r}"
        )
        if domain_ids:
            assert domain_id is not None, f"conversation_concept {concept_id!r} has no macro_domain_id"

    seen_segment_ids: set[str] = set()
    for segment in payload["segments"]:
        segment_id = segment.get("id")
        assert isinstance(segment_id, str) and segment_id, "segment.id must be a non-empty string"
        assert segment_id not in seen_segment_ids, f"duplicate segment id: {segment_id!r}"
        seen_segment_ids.add(segment_id)
        assert isinstance(segment.get("label"), str), "segment.label required (may be empty)"
        assert isinstance(segment.get("chat_id"), str) and segment["chat_id"], "segment.chat_id required"
        assert isinstance(segment.get("platform"), str) and segment["platform"], "segment.platform required"
        for key in ("x", "y", "z"):
            assert isinstance(segment.get(key), (int, float)), f"segment.{key} must be numeric"
        assert isinstance(segment.get("word_count"), int), "segment.word_count must be an int"
        assert isinstance(segment.get("is_tangent"), bool), "segment.is_tangent must be a bool"
        concept_id = segment.get("conversation_concept_id")
        assert concept_id is None or concept_id in concept_ids, (
            f"segment.conversation_concept_id references unknown concept: {concept_id!r}"
        )
        if concept_ids:
            assert concept_id is not None, f"segment {segment_id!r} has no conversation_concept_id"
        domain_id = segment.get("macro_domain_id")
        assert domain_id is None or domain_id in domain_ids, (
            f"segment.macro_domain_id references unknown macro_domain: {domain_id!r}"
        )

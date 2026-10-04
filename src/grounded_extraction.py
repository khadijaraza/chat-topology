"""Grounded, similarity-ranked keyword extraction per segment.

Replaces frequency-count ranking with embedding similarity: a keyword that
best *summarizes* a segment's topic should rank highest, regardless of
whether it appeared once or fifty times. Candidates are still generated
the same way as the original extractor (spaCy noun chunks / entities /
proper nouns, filtered through the same stopword + markup-junk checks --
see src/extract_keywords.py), but instead of counting occurrences, each
distinct candidate is embedded once, ranked by cosine similarity to the
segment's own embedding, and diversified with MMR so near-duplicate
phrases (e.g. "mrt station" / "ximen mrt" / "mrt blue line") don't crowd
out genuinely distinct ones.

Also does corpus-level candidate canonicalization (typo/shorthand
dictionary snapping + corpus-frequency string-similarity snapping -- see
build_canonicalization_map) before ranking, since that needs global term
frequency across every segment, not just the one being processed.

Known limitation: the plan called for stripping fenced code blocks before
candidate generation, but this corpus's ingestion (src/ingest.py's
html_to_text()) already discards <pre>/<code> tag boundaries when
converting HTML to plain text -- there is no fence marker left to strip.
Only LaTeX ($...$ / $$...$$, which survive as literal characters), CLI
flags, and assignment-looking fragments can be stripped at the text level
here; residual code-syntax junk in candidates still relies on
SPECIFICITY_PATTERNS (src/extract_keywords.py) to filter it out downstream.
Properly fixing this means changing ingestion to preserve code-block
boundaries as explicit markers -- a separate, earlier-pipeline change.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.extract_keywords import (
    MULTI_WORD_ENTITY_LABELS,
    content_nouns_in_span,
    get_nlp,
    get_stopwords,
    is_valid_term,
    normalize_phrase,
    normalize_token_lemma,
)
from src.text_embedding import embed_long_text, get_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUNDED_EXTRACTION_CONFIG_PATH = PROJECT_ROOT / "config" / "grounded_extraction.yaml"

# Fallbacks used only if config/grounded_extraction.yaml is missing or malformed.
FALLBACK_CONFIG = {
    "model_name": "all-mpnet-base-v2",
    "max_embedding_tokens": 384,
    "top_k_per_segment": 8,
    "mmr_lambda": 0.6,
    "max_phrase_words": 3,
    "jaccard_hard_threshold": 0.5,
    "jaro_winkler_threshold": 0.88,
    "min_frequency_ratio": 3.0,
    "dictionary_snap_max_frequency": 2,
}

# Trailing lemmas stripped when computing a candidate's canonical root
# tuple for surface-variant collapsing (see collapse_surface_variants) --
# conversational verbs/prepositions that got swept into a noun-chunk span
# but don't change what the phrase is actually about (e.g. "greedy
# algorithms explain" -> same root as "greedy algorithm" once "explain" is
# stripped). Not a config value: this is a fixed linguistic stopword-style
# list, matching how MODIFIER_NOUNS/ARTICLES are hardcoded, not YAML, in
# src/extract_keywords.py.
TRAILING_STRIP_LEMMAS = {
    "explain", "describe", "show", "tell", "discuss", "cover", "please",
    "help", "give", "provide", "ask", "want", "need",
    "in", "on", "at", "for", "to", "of", "with", "about", "into",
}

# A candidate ending in a token this short is almost always a cut-off
# fragment, not a real content-bearing final word (e.g. "kjør i" -- the
# Norwegian preposition "i" survives spaCy's English pipeline as an
# unreliable OOV token, and no language-specific handling is worth adding
# just for this). Language-agnostic: doesn't require knowing what language
# the fragment is in, just that real final nouns are rarely this short.
MIN_FINAL_TOKEN_LENGTH = 2

# Known hyphenated single-letter/short technical compounds that must stay
# atomic. Without this, spaCy's default tokenizer splits e.g. "little-o"
# into ["little", "-", "o"]; the lone "o" then fails length/validity
# checks and gets dropped, leaking the dangling fragment "little-" into
# candidates. Protected at the tokenizer level (_ensure_tokenizer_protections)
# so the split never happens, not just patched up after the fact.
TECHNICAL_HYPHEN_COMPOUNDS = {
    "little-o", "big-o", "big-O", "big-Theta", "big-theta",
    "big-Omega", "big-omega", "n-gram", "n-grams", "p-value", "p-values",
    "t-test", "z-score", "r-squared", "k-means", "k-nearest",
}

# LaTeX delimiters survive real ingestion as literal characters (unlike
# code fences, which don't -- see module docstring).
_LATEX_BLOCK_RE = re.compile(r"\${2}.*?\${2}", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$[^$\n]+\$")
# CLI-style flags: --flag, -f (not a hyphenated word like "well-known").
_CLI_FLAG_RE = re.compile(r"(?<!\w)--?[A-Za-z][\w-]*")
# Assignment-looking fragments: `x = 5`, `foo="bar"` -- not prose like "x is 5".
_ASSIGNMENT_RE = re.compile(r"\b\w+\s*=\s*\S+")

# normalize_phrase() (src/extract_keywords.py) only strips leading a/an/the
# -- other determiners (demonstratives, possessives, interrogatives) still
# leak into noun-chunk candidates as "that malloc", "my c program", "what
# tool". Fixed locally here rather than in the shared function, which the
# already-validated legacy extract_keywords()/concept_clustering() pipeline
# still depends on -- not touching that behavior without reason.
_LEADING_DETERMINER_RE = re.compile(
    r"^(?:this|that|these|those|my|your|his|her|its|our|their|what|which|whose)\s+"
)

# Dangling leading/trailing punctuation or lone operators -- a candidate
# ending in "-" (or starting with one) is almost always a split technical
# compound or stray formatting character, never a real phrase boundary.
_DANGLING_BOUNDARY_RE = re.compile(r"^[-_~+=/]+|[-_~+=/]+$")

_tokenizer_protected = False


def _ensure_tokenizer_protections(nlp) -> None:
    """Register TECHNICAL_HYPHEN_COMPOUNDS as tokenizer special cases so
    they're never split in the first place. Idempotent; cheap to call on
    every generate_candidates() invocation."""
    global _tokenizer_protected
    if _tokenizer_protected:
        return
    for compound in TECHNICAL_HYPHEN_COMPOUNDS:
        nlp.tokenizer.add_special_case(compound, [{"ORTH": compound}])
    _tokenizer_protected = True


def _strip_leading_determiner(term: str) -> str:
    stripped = _LEADING_DETERMINER_RE.sub("", term)
    return stripped if len(stripped) >= 2 else term


def _repair_dangling_boundary(term: str) -> str | None:
    """Strip a leading/trailing punctuation/operator fragment; discard if
    what's left is too short to be a real word. Second line of defense
    behind the tokenizer protection above (covers compounds not in the
    whitelist, or any other source of a dangling boundary character)."""
    if term.lower() in {c.lower() for c in TECHNICAL_HYPHEN_COMPOUNDS}:
        return term
    repaired = _DANGLING_BOUNDARY_RE.sub("", term).strip()
    return repaired if len(repaired) >= 2 else None


def _ends_with_fragment_token(term: str) -> bool:
    words = term.split()
    return bool(words) and len(words[-1]) < MIN_FINAL_TOKEN_LENGTH


@lru_cache(maxsize=1)
def get_grounded_extraction_config(path: str = str(DEFAULT_GROUNDED_EXTRACTION_CONFIG_PATH)) -> dict:
    """Load model/ranking settings from config/grounded_extraction.yaml."""
    config_path = Path(path)
    if not config_path.is_file():
        return dict(FALLBACK_CONFIG)

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return {key: payload.get(key, default) for key, default in FALLBACK_CONFIG.items()}


def strip_non_discursive(text: str) -> str:
    """Strip what can actually be stripped at the text level -- see module
    docstring for why fenced code blocks aren't included."""
    text = _LATEX_BLOCK_RE.sub(" ", text)
    text = _LATEX_INLINE_RE.sub(" ", text)
    text = _ASSIGNMENT_RE.sub(" ", text)
    text = _CLI_FLAG_RE.sub(" ", text)
    return text


def generate_candidates(
    text: str, stopwords: frozenset[str], max_phrase_words: int
) -> dict[str, str | None]:
    """Generate deduped candidate phrases from segment text: named
    entities, noun chunks (already include adjectival modifiers, e.g.
    "neural network"), and standalone proper nouns -- the same candidate
    sources as extract_keywords.py.

    Returns term -> entity_label: the NER label a candidate was extracted
    under (e.g. "WORK_OF_ART"/"PERSON"/"ORG"/"GPE"), or None for
    noun-chunk/PROPN-token candidates with no entity tag. Used downstream
    to bias concept labeling toward specific named entities over generic
    abstract nouns (src/concept_clustering.py's label_concepts_by_specificity).
    """
    nlp = get_nlp()
    _ensure_tokenizer_protections(nlp)
    doc = nlp(text)
    candidates: dict[str, str | None] = {}

    def _add(term: str | None, entity_label: str | None = None) -> None:
        if not term:
            return
        term = _strip_leading_determiner(term)
        repaired = _repair_dangling_boundary(term)
        if repaired is None:
            return
        term = repaired
        if not is_valid_term(term, stopwords):
            return
        if len(term.split()) > max_phrase_words:
            return
        if _ends_with_fragment_token(term):
            return
        if term not in candidates or candidates[term] is None:
            candidates[term] = entity_label

    for ent in doc.ents:
        if ent.label_ in MULTI_WORD_ENTITY_LABELS and len(ent) > 1:
            _add(normalize_phrase(ent), ent.label_)

    for chunk in doc.noun_chunks:
        if not content_nouns_in_span(chunk):
            continue
        _add(normalize_phrase(chunk))

    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            _add(normalize_token_lemma(token))

    return candidates


@lru_cache(maxsize=1)
def _get_spellchecker():
    from spellchecker import SpellChecker

    return SpellChecker(distance=1)


def dictionary_snap(word: str, checker) -> str:
    """Snap a single word to its dictionary form if it's exactly
    edit-distance-1 from a known high-frequency English word (a symmetric-
    delete style lookup, e.g. "thrifte" -> "thrift"). Unchanged if already
    a known word or no distance-1 correction exists."""
    lower = word.lower()
    if not lower.isalpha() or len(lower) < 3 or lower in checker:
        return word
    correction = checker.correction(lower)
    return correction if correction and correction != lower else word


def _jaro_winkler(s1: str, s2: str) -> float:
    """Standard Jaro-Winkler string similarity (pure Python -- avoids
    adding a dependency for one small, well-defined function)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(0, max(len1, len2) // 2 - 1)
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start, end = max(0, i - match_distance), min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3

    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            break
        prefix += 1
        if prefix == 4:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def canonicalize_by_corpus_frequency(
    term_frequencies: dict[str, int], threshold: float, min_frequency_ratio: float
) -> dict[str, str]:
    """For each low-frequency single-word term, if a much more frequent
    (>= min_frequency_ratio times) term elsewhere in the corpus is a close
    string match (Jaro-Winkler >= threshold), remap the rare variant to
    the dominant form (e.g. "thrifte" (1 occurrence) -> "thrifted" (12
    occurrences)). Grouped by first letter first -- typos essentially
    never change the first character, and this keeps the comparison
    tractable on a corpus-sized vocabulary instead of true O(n^2)."""
    by_first_letter: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for term, freq in term_frequencies.items():
        if " " not in term and term.isalpha():
            by_first_letter[term[0].lower()].append((term, freq))

    canonical_map: dict[str, str] = {}
    for group in by_first_letter.values():
        for term, freq in group:
            best_match, best_freq = None, freq
            for other, other_freq in group:
                if other == term or other_freq < freq * min_frequency_ratio:
                    continue
                if other_freq > best_freq and _jaro_winkler(term, other) >= threshold:
                    best_match, best_freq = other, other_freq
            if best_match:
                canonical_map[term] = best_match
    return canonical_map


def build_canonicalization_map(term_frequencies: Counter[str]) -> dict[str, str]:
    """Combine dictionary snapping (typos within edit-distance 1 of a known
    English word) with corpus-frequency canonicalization (a rare variant
    snapped to a much more frequent, string-similar dominant form) into one
    term -> canonical-term map. Scoped to single-word terms only --
    collapse_surface_variants already handles multi-word phrase surface
    variation via lemma-root grouping."""
    cfg = get_grounded_extraction_config()
    checker = _get_spellchecker()

    # Only snap LOW-frequency terms: a general-English dictionary doesn't
    # know domain jargon ("malloc", "kirchhoff", "valgrind"), and will
    # confidently "correct" it to an unrelated real word ("malloc" ->
    # "mallow", found on real data during testing). A typo is inherently
    # rare; real technical vocabulary repeats. A term the corpus itself
    # uses repeatedly is real vocabulary for this corpus regardless of
    # whether a general dictionary recognizes it.
    dictionary_map: dict[str, str] = {}
    for term, freq in term_frequencies.items():
        if " " in term or freq > cfg["dictionary_snap_max_frequency"]:
            continue
        snapped = dictionary_snap(term, checker)
        if snapped != term:
            dictionary_map[term] = snapped

    post_dictionary_frequency: Counter[str] = Counter()
    for term, freq in term_frequencies.items():
        post_dictionary_frequency[dictionary_map.get(term, term)] += freq

    frequency_map = canonicalize_by_corpus_frequency(
        post_dictionary_frequency, cfg["jaro_winkler_threshold"], cfg["min_frequency_ratio"]
    )

    combined: dict[str, str] = {}
    for term in term_frequencies:
        step1 = dictionary_map.get(term, term)
        step2 = frequency_map.get(step1, step1)
        if step2 != term:
            combined[term] = step2
    return combined


def _root_tuple(term: str, nlp) -> tuple[str, ...]:
    """Canonical stem representation: lowercase lemmas, trailing
    conversational verbs/prepositions stripped. "greedy algorithms",
    "greedy algorithm", and "greedy algorithms explain" all reduce to the
    same tuple -- surface variants that should collapse into one candidate
    before they ever reach MMR."""
    doc = nlp(term)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_space and not token.is_punct]
    while lemmas and lemmas[-1] in TRAILING_STRIP_LEMMAS:
        lemmas.pop()
    return tuple(lemmas)


def collapse_surface_variants(candidates: list[str]) -> list[str]:
    """Group candidates by canonical root tuple; keep one representative
    per group (the shortest surface form -- the cleanest/most natural
    variant, not one padded with a trailing filler word). Runs before
    embedding, since the whole point is to embed each *distinct* candidate
    once rather than several near-identical surface forms of the same one."""
    nlp = get_nlp()
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for term in candidates:
        root = _root_tuple(term, nlp)
        if root:
            groups[root].append(term)

    return [min(members, key=lambda m: (len(m.split()), len(m))) for members in groups.values()]


def _token_jaccard(a: str, b: str) -> float:
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def mmr_select(
    candidates: list[str],
    candidate_vectors: np.ndarray,
    segment_vector: np.ndarray,
    k: int,
    lambda_param: float,
    jaccard_hard_threshold: float,
) -> list[tuple[str, float]]:
    """Maximal Marginal Relevance: iteratively pick the candidate that
    maximizes (lambda * relevance - (1 - lambda) * penalty), where penalty
    is the max, over already-picked candidates, of
    max(cosine_similarity, token_jaccard_overlap) -- embedding similarity
    alone can under-penalize phrases that share most of their words but
    embed slightly differently ("heap allocator metadata" vs. "heap
    allocator"). Overlap at or above jaccard_hard_threshold forces the
    penalty to 1.0 (an automatic near-veto), matching how blatant a
    shared-token duplicate is. Returns up to k (term, relevance_score)
    pairs -- relevance_score is plain cosine similarity to the segment,
    not the MMR score itself (which also factors in diversity and isn't
    meaningful to compare across segments)."""
    if not candidates:
        return []

    relevance = candidate_vectors @ segment_vector  # cosine sim, vectors are unit-normalized
    selected: list[int] = [int(np.argmax(relevance))]
    remaining = [i for i in range(len(candidates)) if i != selected[0]]

    def penalty(idx: int, other: int) -> float:
        jaccard = _token_jaccard(candidates[idx], candidates[other])
        if jaccard >= jaccard_hard_threshold:
            return 1.0
        cos_sim = float(candidate_vectors[idx] @ candidate_vectors[other])
        return max(cos_sim, jaccard)

    while remaining and len(selected) < k:
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            max_penalty = max(penalty(idx, s) for s in selected)
            mmr_score = lambda_param * relevance[idx] - (1 - lambda_param) * max_penalty
            if mmr_score > best_score:
                best_idx, best_score = idx, mmr_score
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [(candidates[i], float(relevance[i])) for i in selected]


def prune_substrings(selected: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Post-MMR safety net: if phrase A is an exact substring of phrase B,
    drop B (the longer, more conversational one) and keep A. Catches
    anything collapse_surface_variants + the Jaccard penalty didn't
    (e.g. a novel trailing word not in TRAILING_STRIP_LEMMAS)."""
    terms = [term for term, _ in selected]
    drop: set[int] = set()
    for i, term_i in enumerate(terms):
        for j, term_j in enumerate(terms):
            if i != j and term_i != term_j and term_i in term_j and j not in drop:
                drop.add(j)
    return [item for idx, item in enumerate(selected) if idx not in drop]


def rank_candidates(
    segment_vector: np.ndarray, candidates: dict[str, str | None], model, config: dict
) -> list[dict]:
    """Shared ranking tail: collapse surface variants -> embed -> MMR ->
    prune substrings. Takes the segment's own embedding as a precomputed
    input (rather than the raw text) so callers that already need it for
    other purposes -- e.g. the proximity-layout pipeline persisting every
    segment's embedding -- don't pay for it twice. Returns a ranked list
    of {"term", "score", "entity_label"} (score = cosine similarity to the
    segment, replacing "count" as the ranking signal; entity_label carried
    through from generate_candidates for Phase 3's specificity-weighted
    labeling)."""
    terms = sorted(candidates)
    collapsed = collapse_surface_variants(terms)
    if not collapsed:
        return []

    candidate_vectors = model.encode(collapsed, normalize_embeddings=True, show_progress_bar=False)

    ranked = mmr_select(
        collapsed,
        candidate_vectors,
        segment_vector,
        config["top_k_per_segment"],
        config["mmr_lambda"],
        config["jaccard_hard_threshold"],
    )
    ranked = prune_substrings(ranked)
    ranked.sort(key=lambda item: -item[1])
    return [
        {"term": term, "score": score, "entity_label": candidates.get(term)} for term, score in ranked
    ]


def extract_grounded_keywords(segment_text: str, model, config: dict) -> list[dict]:
    """Single-segment convenience entry point: clean -> generate candidates
    -> rank. No corpus-level canonicalization context (that needs global
    term frequency across every segment) -- see extract_segment_keywords
    for the real corpus pipeline."""
    stopwords = get_stopwords()
    cleaned = strip_non_discursive(segment_text)
    candidates = generate_candidates(cleaned, stopwords, config["max_phrase_words"])
    segment_vector = embed_long_text(cleaned, model, config["max_embedding_tokens"])
    return rank_candidates(segment_vector, candidates, model, config)


def extract_segment_keywords(
    conversations: list[dict], segments_by_chat: dict[str, list[dict]]
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Extract grounded keywords for every segment across every chat.

    Two passes: (1) generate raw candidates per segment once (the
    expensive spaCy work) while accumulating global term frequency, then
    build a corpus-wide canonicalization map (build_canonicalization_map);
    (2) apply that map and do the embed/rank/MMR/prune tail per segment.
    Splitting it this way means the spaCy candidate generation only runs
    once per segment, not twice, despite canonicalization needing global
    context that isn't available until every segment's raw candidates
    have been seen.

    Returns (results, segment_embeddings):
    - results: a list of {"segment_id", "chat_id", "label", "keywords": [...]}
      entries -- one per segment, same shape convention as chat_keywords.json
      but keyed by segment instead of whole chat.
    - segment_embeddings: {segment_id: embedding}, every segment with
      non-empty text (independent of whether it produced any keyword
      candidates) -- callers doing segment-level clustering/projection
      (src/conversation_concepts.py, src/proximity_layout.py) need a
      position for every segment, not just ones with surviving keywords,
      and this avoids re-embedding text already embedded once here.
    """
    config = get_grounded_extraction_config()
    model = get_embedding_model(config["model_name"])
    stopwords = get_stopwords()

    turns_by_chat = {conv["chat_id"]: conv.get("turns", []) for conv in conversations if conv.get("chat_id")}

    segment_order: list[tuple[str, str, str]] = []
    cleaned_by_segment: dict[str, str] = {}
    candidates_by_segment: dict[str, dict[str, str | None]] = {}
    global_frequency: Counter[str] = Counter()

    for chat_id, segments in segments_by_chat.items():
        turns = turns_by_chat.get(chat_id, [])
        for segment in segments:
            segment_id = segment["segment_id"]
            segment_turns = turns[segment["start_turn"]:segment["end_turn"]]
            text = "\n".join(
                turn["text"] for turn in segment_turns if isinstance(turn.get("text"), str)
            )
            cleaned = strip_non_discursive(text) if text else ""
            candidates = (
                generate_candidates(cleaned, stopwords, config["max_phrase_words"]) if cleaned else {}
            )

            segment_order.append((chat_id, segment_id, segment["label"]))
            cleaned_by_segment[segment_id] = cleaned
            candidates_by_segment[segment_id] = candidates
            for term in candidates:
                global_frequency[term] += 1

    canonical_map = build_canonicalization_map(global_frequency)

    results: list[dict] = []
    segment_embeddings: dict[str, np.ndarray] = {}
    for chat_id, segment_id, label in segment_order:
        cleaned = cleaned_by_segment[segment_id]
        segment_vector = embed_long_text(cleaned, model, config["max_embedding_tokens"]) if cleaned else None
        if segment_vector is not None:
            segment_embeddings[segment_id] = segment_vector

        raw_candidates = candidates_by_segment[segment_id]
        canonical_candidates: dict[str, str | None] = {}
        for term, entity_label in raw_candidates.items():
            canonical_term = canonical_map.get(term, term)
            if canonical_term not in canonical_candidates or canonical_candidates[canonical_term] is None:
                canonical_candidates[canonical_term] = entity_label

        keywords = (
            rank_candidates(segment_vector, canonical_candidates, model, config)
            if canonical_candidates and segment_vector is not None
            else []
        )
        results.append({"segment_id": segment_id, "chat_id": chat_id, "label": label, "keywords": keywords})

    return results, segment_embeddings

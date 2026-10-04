"""Extract condensed topical keywords from normalized conversations using spaCy.

Mixed file, stays in src/ root rather than moving to src/legacy/: several
NLP helpers here (get_nlp, get_stopwords, is_valid_term, normalize_phrase,
normalize_token_lemma, content_nouns_in_span, MULTI_WORD_ENTITY_LABELS) are
shared -- src/grounded_extraction.py (current pipeline) imports them
directly rather than duplicating spaCy pipeline setup, stopword filtering,
and SPECIFICITY_PATTERNS-based validity checks. extract_keywords() itself
(the whole-chat, frequency-ranked entrypoint that ties those helpers
together) is legacy-only, called only by scripts/legacy/run_extraction.py
and scripts/legacy/run_build_output.py -- superseded by
src/grounded_extraction.py's per-segment, similarity-ranked extraction.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import spacy
import yaml
from spacy.language import Language

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STOPWORDS_PATH = PROJECT_ROOT / "config" / "keyword_stopwords.yaml"
DEFAULT_PHRASES_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "phrases.json"
DEFAULT_ALIASES_PATH = PROJECT_ROOT / "config" / "keyword_aliases.yaml"

MULTI_WORD_ENTITY_LABELS = {
    "GPE",
    "ORG",
    "PERSON",
    "NORP",
    "FAC",
    "PRODUCT",
    "WORK_OF_ART",
    "EVENT",
    "LAW",
    "LANGUAGE",
}

ARTICLES = {"a", "an", "the"}
POSSESSIVE_RE = re.compile(r"(?:'s|'s)$|'$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
WORD_BOUNDARY_RE = re.compile(r"\b[\w'-]+\b")

# Drop overly specific or structural nouns that rarely define a topic on their own.
MODIFIER_NOUNS = {
    "stop",
    "station",
    "schedule",
    "minute",
    "hour",
    "time",
    "price",
    "rating",
    "total",
    "level",
    "side",
    "end",
    "start",
    "rest",
    "form",
    "case",
    "state",
    "method",
    "path",
    "area",
    "goal",
    "context",
    "concept",
    "difference",
    "table",
    "error",
    "standard",
    "series",
    "sum",
    "student",
    "final",
    "walk",
    "ride",
    "travel",
    "stay",
    "deal",
    "night",
    "morning",
    "evening",
    "afternoon",
    "number",
    "amount",
    "size",
    "type",
    "kind",
    "part",
    "section",
    "line",
    "page",
    "item",
    "option",
    "step",
    "point",
    "example",
    "version",
    "update",
    "change",
    "detail",
    "note",
    "minute",
    "second",
    "week",
    "month",
    "year",
    "day",
    "bed",
    "room",
    "ticket",
    "reservation",
    "route",
    "direction",
    "distance",
    "duration",
    "cost",
    "fee",
}

SPECIFICITY_PATTERNS = (
    re.compile(r"\$"),
    re.compile(r"\d:\d"),  # clock times
    re.compile(r"\b(?:am|pm)\b"),
    re.compile(r"^\d"),
    re.compile(r"\d+\s*(?:cm|mm|km|mi|ft|lb|kg|mph|kph)\b"),
    re.compile(r"^\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*$"),  # mostly numeric
    re.compile(r"^[\d$./:-]+$"),
    re.compile(r"[_+]"),  # code / compound tokens
    re.compile(r"https?://"),
    # HTML/LaTeX/code markup leaking through from pasted code or document
    # snippets in a chat (e.g. `div class="box`, `\vspace{0.5cm`,
    # `print(f"{time`, `</script`). The `\` and `^\d` checks above only
    # catch a backslash/digit at the very START of a term -- spaCy noun
    # chunks routinely start mid-expression, so `n/3 \rceil` or
    # `sequential([\n` slip through unless matched anywhere in the string.
    re.compile(r"\\"),  # backslash anywhere: LaTeX commands, Windows paths, escapes
    re.compile(r"[{}<>]"),  # LaTeX groups / dict-set literals / HTML tags
    re.compile(r'["`]'),  # code-style string quoting
    re.compile(r"[\[\]]"),  # code array/index literals
    # Found on real segment-level extraction (Phase 2/3 of the pipeline
    # reconstruction): function-call fragments like `pish(stdin`,
    # `realloc(word` slipped through since literal parentheses were never
    # in this list. Natural-language noun phrases essentially never
    # contain "(" or ")" -- unlike underscores/brackets/braces above,
    # there's no legitimate prose case this would over-match.
    re.compile(r"[()]"),
)


class KeywordHit(TypedDict):
    term: str
    count: int


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def get_stopwords(path: str = str(DEFAULT_STOPWORDS_PATH)) -> frozenset[str]:
    stopwords_path = Path(path)
    words: set[str] = set(MODIFIER_NOUNS)
    if stopwords_path.is_file():
        with stopwords_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        for word in payload.get("stopwords", []):
            if isinstance(word, str) and word.strip():
                words.add(word.strip().lower())
    words.update(get_nlp().Defaults.stop_words)
    return frozenset(words)


@lru_cache(maxsize=1)
def get_phrase_lookup(path: str = str(DEFAULT_PHRASES_PATH)) -> tuple[dict[tuple[str, ...], str], int]:
    """Load corpus-level phrases detected by src/legacy/phrase_detection.py.

    Returns (lookup, max_phrase_len): lookup maps a tuple of consecutive
    lemma words -> the canonical phrase string to emit for them (e.g.
    ("machine", "learning") -> "machine learning"). If phrases.json hasn't
    been generated yet (scripts/legacy/run_phrase_detection.py hasn't run), this
    degrades gracefully to an empty lookup -- extraction still works, it
    just won't merge multi-word compounds beyond what spaCy's own NER
    entities give it.
    """
    phrases_path = Path(path)
    lookup: dict[tuple[str, ...], str] = {}
    if not phrases_path.is_file():
        return lookup, 0

    with phrases_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    max_len = 0
    for hit in payload:
        words = hit.get("words")
        phrase = hit.get("phrase")
        if not isinstance(words, list) or not isinstance(phrase, str):
            continue
        key = tuple(str(w).strip().lower() for w in words if str(w).strip())
        if not key:
            continue
        lookup[key] = phrase
        max_len = max(max_len, len(key))

    return lookup, max_len


@lru_cache(maxsize=1)
def _parse_alias_config(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse config/keyword_aliases.yaml into (word_aliases, term_aliases).

    word_aliases: single-word -> single-word entries (e.g. "github" ->
      "git"). These get applied as early as possible -- right after
      lemmatization, before phrase detection or extraction ever counts
      anything -- so that e.g. "github" and "git" tokens are unified BEFORE
      phrase formation. Applying this only as a final whole-term
      substitution isn't enough: phrase detection would already have locked
      in "github repo" as its own separate compound from "git repo" by
      then, and a term-level substitution can't un-merge those back
      together.

    term_aliases: everything else (alias and canonical have different word
      counts, e.g. an acronym like "ucsd" -> "uc san diego"). These can't
      be applied token-for-token, so they're applied once at the end, on
      fully-formed keyword terms.
    """
    aliases_path = Path(path)
    word_aliases: dict[str, str] = {}
    term_aliases: dict[str, str] = {}
    if not aliases_path.is_file():
        return word_aliases, term_aliases

    with aliases_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    groups = payload.get("aliases", {})
    if not isinstance(groups, dict):
        return word_aliases, term_aliases

    seen_as_canonical: set[str] = set()
    all_aliases: dict[str, str] = {}
    for canonical, alias_list in groups.items():
        if not isinstance(canonical, str):
            continue
        canonical_norm = WHITESPACE_RE.sub(" ", canonical.strip().lower())
        if not canonical_norm or not isinstance(alias_list, list):
            continue
        seen_as_canonical.add(canonical_norm)
        for alias in alias_list:
            if not isinstance(alias, str):
                continue
            alias_norm = WHITESPACE_RE.sub(" ", alias.strip().lower())
            if not alias_norm or alias_norm == canonical_norm:
                continue
            if alias_norm in all_aliases and all_aliases[alias_norm] != canonical_norm:
                raise ValueError(
                    f"config/keyword_aliases.yaml: '{alias_norm}' is mapped to both "
                    f"'{all_aliases[alias_norm]}' and '{canonical_norm}' -- fix the conflict."
                )
            all_aliases[alias_norm] = canonical_norm
            if " " not in alias_norm and " " not in canonical_norm:
                word_aliases[alias_norm] = canonical_norm
            else:
                term_aliases[alias_norm] = canonical_norm

    conflicts = seen_as_canonical & set(all_aliases.keys())
    if conflicts:
        raise ValueError(
            f"config/keyword_aliases.yaml: {sorted(conflicts)} are each listed as both "
            "a canonical term and an alias of something else -- fix the conflict."
        )

    return word_aliases, term_aliases


def get_word_aliases(path: str = str(DEFAULT_ALIASES_PATH)) -> dict[str, str]:
    return _parse_alias_config(path)[0]


def get_term_aliases(path: str = str(DEFAULT_ALIASES_PATH)) -> dict[str, str]:
    return _parse_alias_config(path)[1]


def apply_word_alias(lemma: str, word_aliases: dict[str, str] | None = None) -> str:
    """Canonicalize a single lemma (e.g. "github" -> "git") if it's a known
    single-word alias. Safe to call with no lookup built yet -- resolves
    the default config lazily."""
    lookup = word_aliases if word_aliases is not None else get_word_aliases()
    return lookup.get(lemma, lemma)


def apply_term_aliases(counts: Counter[str], term_aliases: dict[str, str]) -> Counter[str]:
    """Merge whole-term aliases (different word counts, e.g. an acronym)
    into their canonical term's count. Word-level aliases should already be
    resolved by this point (see apply_word_alias), so this only needs to
    handle the cases that couldn't be -- but it's applied defensively to
    every term either way, in case a term reached here without going
    through per-token alias resolution."""
    merged: Counter[str] = Counter()
    for term, count in counts.items():
        canonical = term_aliases.get(term, term)
        merged[canonical] += count
    return merged


def segment_with_phrases(
    nouns: list[str],
    phrase_lookup: dict[tuple[str, ...], str],
    max_phrase_len: int,
) -> list[str]:
    """Greedily replace runs of adjacent nouns with known corpus-level
    phrases (longest match first); unmatched nouns pass through as
    individual terms.

    Each noun position contributes to exactly one output term -- either as
    part of a matched compound, or as itself -- never both. That's the fix
    for the double-counting bug: previously a chunk like "machine learning"
    incremented "machine", "learning", AND "machine learning" from the same
    span, which guarantees near-perfect PMI between "machine" and
    "learning" since they're extracted together almost every time by
    construction, not because they're independently correlated.
    """
    if max_phrase_len < 2:
        return list(nouns)

    terms: list[str] = []
    i = 0
    n = len(nouns)
    while i < n:
        matched = False
        upper = min(max_phrase_len, n - i)
        for length in range(upper, 1, -1):
            candidate = tuple(nouns[i : i + length])
            phrase = phrase_lookup.get(candidate)
            if phrase is not None:
                terms.append(phrase)
                i += length
                matched = True
                break
        if not matched:
            terms.append(nouns[i])
            i += 1
    return terms


def conversation_text(conversation: dict) -> str:
    turns = conversation.get("turns", [])
    chunks: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        text = turn.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks)


def normalize_token_lemma(token, word_aliases: dict[str, str] | None = None) -> str | None:
    if token.is_space or token.is_punct:
        return None
    if token.pos_ not in {"NOUN", "PROPN"}:
        return None
    lemma = token.lemma_.lower().strip()
    lemma = POSSESSIVE_RE.sub("", lemma)
    if len(lemma) < 2:
        return None
    # Resolve single-word aliases (e.g. "github" -> "git") here, as early as
    # possible -- before this lemma ever gets used to form a multi-word
    # phrase. See _parse_alias_config for why this can't wait until the end.
    return apply_word_alias(lemma, word_aliases)


def normalize_phrase(tokens) -> str | None:
    parts: list[str] = []
    for token in tokens:
        if token.is_space or token.is_punct:
            continue
        if token.pos_ == "DET" and token.lemma_.lower() in ARTICLES:
            continue
        if token.pos_ == "PART" and token.lemma_.lower() == "'s":
            continue
        if token.text in {"'s", "'"}:
            continue
        lemma = token.lemma_.lower().strip()
        lemma = POSSESSIVE_RE.sub("", lemma)
        if not lemma or lemma in ARTICLES:
            continue
        parts.append(apply_word_alias(lemma))

    if not parts:
        return None

    term = WHITESPACE_RE.sub(" ", " ".join(parts)).strip()
    term = POSSESSIVE_RE.sub("", term)
    if len(term) < 2:
        return None
    return term


def is_too_specific(term: str) -> bool:
    if term.startswith("\\"):
        return True
    if term.isdigit():
        return True
    digit_ratio = sum(ch.isdigit() for ch in term) / max(len(term), 1)
    if digit_ratio > 0.35:
        return True
    return any(pattern.search(term) for pattern in SPECIFICITY_PATTERNS)


def is_valid_term(term: str, stopwords: frozenset[str]) -> bool:
    if not term or len(term) < 2:
        return False
    if term in stopwords:
        return False
    if is_too_specific(term):
        return False
    return True


def add_count(counts: Counter[str], term: str | None, stopwords: frozenset[str], weight: int = 1) -> None:
    if not term:
        return
    term = WHITESPACE_RE.sub(" ", term.strip().lower())
    term = POSSESSIVE_RE.sub("", term)
    term = re.sub(r"^(?:the|a|an)\s+", "", term)
    if not is_valid_term(term, stopwords):
        return
    counts[term] += weight


def content_nouns_in_span(span) -> list[str]:
    nouns: list[str] = []
    for token in span:
        lemma = normalize_token_lemma(token)
        if lemma:
            nouns.append(lemma)
    return nouns


def lemma_for_single_word(term: str, nlp: Language) -> str:
    doc = nlp(term)
    if not doc:
        return term
    return doc[0].lemma_.lower()


def merge_lemma_variants(counts: Counter[str], nlp: Language) -> Counter[str]:
    """Collapse singular/plural variants for single-word topical terms."""
    merged = Counter(counts)
    singles = [term for term in merged if " " not in term]
    lemma_groups: dict[str, list[str]] = {}
    for term in singles:
        lemma = lemma_for_single_word(term, nlp)
        lemma_groups.setdefault(lemma, []).append(term)

    for lemma, variants in lemma_groups.items():
        if len(variants) <= 1:
            continue
        canonical = lemma if lemma in variants else min(variants, key=len)
        total = sum(merged[v] for v in variants)
        for variant in variants:
            del merged[variant]
        merged[canonical] = total

    return merged


def condense_keyword_counts(counts: Counter[str], nlp: Language) -> Counter[str]:
    """Collapse trivial lemma variants (singular/plural) and drop low-signal
    bare modifier nouns that survived only because they weren't part of a
    detected multi-word phrase.

    NOTE: earlier versions of this function also folded ANY multi-word
    phrase into whichever of its component words was shortest (e.g.
    "machine learning" -> "machine"). That's removed. It assumed a phrase
    is always just a variant of one of its words, which isn't true -- and
    combined with the phrase/component double-counting bug that used to
    exist in extract_keywords(), it was the main source of the spurious
    "machine" <-> "learning" correlation: the phrase-fold step destroyed
    "machine learning" as its own concept and left an inflated, lopsided
    trail behind. Phrase identity is now decided once, up front, by
    corpus-level phrase detection (src/phrase_detection.py) via
    segment_with_phrases() above, so it shouldn't be silently undone here.
    """
    merged = Counter(counts)
    for term in list(merged.keys()):
        if " " not in term and term in MODIFIER_NOUNS and merged[term] <= 1:
            del merged[term]
    return merge_lemma_variants(merged, nlp)


def counts_to_ranked_hits(counts: Counter[str]) -> list[KeywordHit]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"term": term, "count": count} for term, count in ranked]


def extract_keywords(
    conversation: dict,
    *,
    stopwords_path: Path | None = None,
    phrases_path: Path | None = None,
    aliases_path: Path | None = None,
) -> list[KeywordHit]:
    """Extract condensed topical keywords for one conversation, ranked by occurrence."""
    text = conversation_text(conversation)
    if not text:
        return []

    stopwords = get_stopwords(str(stopwords_path or DEFAULT_STOPWORDS_PATH))
    phrase_lookup, max_phrase_len = get_phrase_lookup(str(phrases_path or DEFAULT_PHRASES_PATH))
    term_aliases = get_term_aliases(str(aliases_path or DEFAULT_ALIASES_PATH))
    nlp = get_nlp()
    doc = nlp(text)
    counts: Counter[str] = Counter()

    for ent in doc.ents:
        nouns = content_nouns_in_span(ent)
        if not nouns:
            continue

        if ent.label_ in MULTI_WORD_ENTITY_LABELS and len(ent) > 1:
            phrase = normalize_phrase(ent)
            if phrase:
                add_count(counts, phrase, stopwords)
                continue  # phrase covers this span; don't also add its component nouns

        for term in segment_with_phrases(nouns, phrase_lookup, max_phrase_len):
            add_count(counts, term, stopwords)

    for chunk in doc.noun_chunks:
        nouns = content_nouns_in_span(chunk)
        if not nouns:
            continue

        for term in segment_with_phrases(nouns, phrase_lookup, max_phrase_len):
            # Preserve the previous behavior of skipping bare structural
            # modifier nouns (e.g. "type", "amount") when they didn't get
            # absorbed into a real phrase -- but never skip a multi-word
            # phrase itself.
            if " " not in term and term in MODIFIER_NOUNS:
                continue
            add_count(counts, term, stopwords)

    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            add_count(counts, normalize_token_lemma(token), stopwords)

    condensed = condense_keyword_counts(counts, nlp)
    canonicalized = apply_term_aliases(condensed, term_aliases)
    return counts_to_ranked_hits(canonicalized)

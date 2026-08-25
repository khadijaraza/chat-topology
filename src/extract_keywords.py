"""Extract condensed topical keywords from normalized conversations using spaCy."""

from __future__ import annotations

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


def normalize_token_lemma(token) -> str | None:
    if token.is_space or token.is_punct:
        return None
    if token.pos_ not in {"NOUN", "PROPN"}:
        return None
    lemma = token.lemma_.lower().strip()
    lemma = POSSESSIVE_RE.sub("", lemma)
    if len(lemma) < 2:
        return None
    return lemma


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
        parts.append(lemma)

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
    """Merge longer phrases into shorter shared topical roots within one chat."""
    merged = Counter(counts)
    terms = sorted(merged.keys(), key=lambda term: (len(term.split()), len(term), -merged[term]))

    for short in terms:
        if short not in merged:
            continue
        short_words = short.split()
        if len(short_words) != 1:
            continue

        for long in list(merged.keys()):
            if long == short:
                continue
            long_words = long.split()
            if len(long_words) <= 1:
                continue
            if short in long_words:
                merged[short] += merged[long]
                del merged[long]

    # Drop single-token modifier nouns when a related multi-word term was already absorbed.
    for term in list(merged.keys()):
        if term in MODIFIER_NOUNS and merged[term] <= 1:
            del merged[term]

    return merge_lemma_variants(merged, nlp)


def counts_to_ranked_hits(counts: Counter[str]) -> list[KeywordHit]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"term": term, "count": count} for term, count in ranked]


def extract_keywords(
    conversation: dict,
    *,
    stopwords_path: Path | None = None,
) -> list[KeywordHit]:
    """Extract condensed topical keywords for one conversation, ranked by occurrence."""
    text = conversation_text(conversation)
    if not text:
        return []

    stopwords = get_stopwords(str(stopwords_path or DEFAULT_STOPWORDS_PATH))
    nlp = get_nlp()
    doc = nlp(text)
    counts: Counter[str] = Counter()

    for ent in doc.ents:
        if ent.label_ in MULTI_WORD_ENTITY_LABELS and len(ent) > 1:
            phrase = normalize_phrase(ent)
            add_count(counts, phrase, stopwords)
        for noun in content_nouns_in_span(ent):
            add_count(counts, noun, stopwords)

    for chunk in doc.noun_chunks:
        nouns = content_nouns_in_span(chunk)
        if not nouns:
            continue

        # Prefer topical nouns over structural modifiers inside a phrase.
        topical = [noun for noun in nouns if noun not in MODIFIER_NOUNS]
        targets = topical or nouns[:1]
        for noun in targets:
            add_count(counts, noun, stopwords)

        if len(nouns) > 1:
            phrase = normalize_phrase(chunk)
            add_count(counts, phrase, stopwords, weight=1)

    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop:
            add_count(counts, normalize_token_lemma(token), stopwords)

    condensed = condense_keyword_counts(counts, nlp)
    return counts_to_ranked_hits(condensed)

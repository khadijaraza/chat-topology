"""Corpus-level phrase (compound keyword) detection via normalized PMI.

This answers a different question than src/extract_keywords.py's spaCy
noun-chunk/entity heuristics. Those ask "does this span look like a noun
phrase in this one sentence" -- a local, syntactic judgment. This module
asks "do these two words appear adjacent to each other, across the WHOLE
corpus, far more often than their individual frequencies would predict" --
a corpus-level, statistical judgment. That makes it robust to real domain
compounds spaCy's NER doesn't recognize (e.g. "prompt engineering"), and
immune to one-off adjacency that happens to occur in a single sentence.

Algorithm (the standard "word2phrase" / gensim Phrases approach): compute
adjacent-bigram NPMI across the corpus; accept pairs that clear both a
minimum count and a minimum NPMI threshold; merge accepted pairs into
single tokens; repeat against the now-partially-merged token stream to
discover longer phrases (e.g. "machine_learning" + "model" ->
"machine_learning_model"), up to max_phrase_words passes.

Run via scripts/run_phrase_detection.py to produce data/processed/phrases.json,
which src/extract_keywords.py then loads to merge detected compounds into
single keywords during extraction (see segment_with_phrases there).
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, Union

import yaml

from src.extract_keywords import apply_word_alias, get_nlp, get_stopwords

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "phrase_detection.yaml"

DEFAULT_MIN_PAIR_COUNT = 5
DEFAULT_MIN_NPMI = 0.55
DEFAULT_MAX_PHRASE_WORDS = 3
DEFAULT_REDUNDANCY_RATIO = 0.85

CONTENT_POS = {"NOUN", "PROPN"}

# A sequence item is either a single lemma, a tuple of lemmas (a merged
# compound from an earlier pass), or None (a break in adjacency -- a
# stopword, punctuation, or non-content token sits here in the original
# text, so nothing should bridge across it into a fake phrase).
Token = Union[str, tuple, None]


class PhraseConfig(TypedDict):
    min_pair_count: int
    min_npmi: float
    max_phrase_words: int
    redundancy_ratio: float


class PhraseHit(TypedDict):
    phrase: str
    words: list[str]
    count: int
    npmi: float


@lru_cache(maxsize=1)
def get_phrase_config(path: str = str(DEFAULT_CONFIG_PATH)) -> PhraseConfig:
    config_path = Path(path)
    payload: dict = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    return {
        "min_pair_count": int(payload.get("min_pair_count", DEFAULT_MIN_PAIR_COUNT)),
        "min_npmi": float(payload.get("min_npmi", DEFAULT_MIN_NPMI)),
        "max_phrase_words": int(payload.get("max_phrase_words", DEFAULT_MAX_PHRASE_WORDS)),
        "redundancy_ratio": float(payload.get("redundancy_ratio", DEFAULT_REDUNDANCY_RATIO)),
    }


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


def content_lemma_sequence(text: str, stopwords: frozenset[str]) -> list[str | None]:
    """One entry per content token: its lemma, or None where a non-content
    token (stopword/punctuation/other POS) breaks adjacency."""
    nlp = get_nlp()
    doc = nlp(text)
    sequence: list[str | None] = []
    for token in doc:
        if token.is_space:
            continue
        if token.pos_ not in CONTENT_POS or token.is_punct:
            sequence.append(None)
            continue
        lemma = token.lemma_.lower().strip()
        if not lemma or lemma in stopwords:
            sequence.append(None)
            continue
        # Resolve single-word aliases (config/keyword_aliases.yaml) BEFORE
        # counting bigrams -- so "git" and "github" tokens are unified from
        # the start, and "git repo" / "github repo" mentions combine into
        # one detected phrase instead of two separate ones that a later
        # whole-term substitution couldn't merge back together.
        sequence.append(apply_word_alias(lemma))
    return sequence


def _words_of(token: Token) -> tuple[str, ...]:
    if token is None:
        return ()
    if isinstance(token, tuple):
        return token
    return (token,)


def count_unigrams_and_bigrams(
    sequences: list[list[Token]],
) -> tuple[Counter[Token], Counter[tuple[Token, Token]], int]:
    unigrams: Counter[Token] = Counter()
    bigrams: Counter[tuple[Token, Token]] = Counter()
    total = 0
    for sequence in sequences:
        prev: Token = None
        for token in sequence:
            if token is not None:
                unigrams[token] += 1
                total += 1
                if prev is not None:
                    bigrams[(prev, token)] += 1
            prev = token
    return unigrams, bigrams, total


def npmi(pair_count: int, count_a: int, count_b: int, total: int) -> float:
    """Normalized PMI in [-1, 1]. 1 = the two always occur together (and
    only together); 0 = independent; negative = anti-correlated."""
    if total == 0 or pair_count == 0 or count_a == 0 or count_b == 0:
        return -1.0
    p_a = count_a / total
    p_b = count_b / total
    p_ab = pair_count / total
    pmi = math.log2(p_ab / (p_a * p_b))
    denom = -math.log2(p_ab)
    if denom == 0:
        return 1.0
    return pmi / denom


def merge_sequence(sequence: list[Token], accepted_pairs: set[tuple[Token, Token]]) -> list[Token]:
    """Merge accepted adjacent (a, b) pairs into a single compound token,
    scanning left to right and non-overlapping (greedy, first match wins)."""
    merged: list[Token] = []
    i = 0
    n = len(sequence)
    while i < n:
        if i + 1 < n and (sequence[i], sequence[i + 1]) in accepted_pairs:
            merged.append(_words_of(sequence[i]) + _words_of(sequence[i + 1]))
            i += 2
        else:
            merged.append(sequence[i])
            i += 1
    return merged


def build_lemma_sequences(
    conversations: list[dict],
    *,
    stopwords_path: Path | None = None,
) -> list[list[Token]]:
    stopwords = get_stopwords(str(stopwords_path)) if stopwords_path else get_stopwords()
    sequences: list[list[Token]] = []
    for conversation in conversations:
        text = conversation_text(conversation)
        if not text:
            continue
        sequences.append(content_lemma_sequence(text, stopwords))
    return sequences


def is_contiguous_subsequence(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    """True if `shorter` appears as a contiguous run anywhere inside `longer`.

    Handles containment on either side: a shorter phrase can end up as a
    prefix of a longer one (e.g. "uc san" -> "uc san diego", built by
    appending a word on the right during chaining) or as a suffix (e.g.
    "san diego" -> "uc san diego", accepted independently in an earlier
    pass before the longer phrase existed). We don't assume which.
    """
    if len(shorter) >= len(longer):
        return False
    span = len(shorter)
    for start in range(len(longer) - span + 1):
        if longer[start : start + span] == shorter:
            return True
    return False


def prune_redundant_phrases(hits: list[PhraseHit], *, redundancy_ratio: float) -> list[PhraseHit]:
    """Drop a shorter phrase when a longer accepted phrase contains it and
    accounts for almost all of its occurrences (see redundancy_ratio in
    config/phrase_detection.yaml). Phrases that also occur independently of
    any longer phrase are kept, even if they're sometimes a sub-span of one.
    """
    by_length = sorted(hits, key=lambda hit: len(hit["words"]))
    kept: list[PhraseHit] = []

    for hit in by_length:
        words = tuple(hit["words"])
        redundant = False
        for other in hits:
            other_words = tuple(other["words"])
            if len(other_words) <= len(words):
                continue
            if not is_contiguous_subsequence(words, other_words):
                continue
            if other["count"] <= 0 or hit["count"] <= 0:
                continue
            if other["count"] / hit["count"] >= redundancy_ratio:
                redundant = True
                break
        if not redundant:
            kept.append(hit)

    return sorted(kept, key=lambda h: (-h["count"], h["phrase"]))


def detect_phrases(
    conversations: list[dict],
    *,
    config: PhraseConfig | None = None,
    stopwords_path: Path | None = None,
) -> list[PhraseHit]:
    """Detect corpus-level compound keywords, iterating up to
    max_phrase_words passes to build longer phrases from accepted shorter
    ones. Returns hits sorted by count (most-evidenced compounds first).
    """
    cfg = config or get_phrase_config()
    sequences = build_lemma_sequences(conversations, stopwords_path=stopwords_path)

    all_phrases: dict[str, PhraseHit] = {}
    passes = max(cfg["max_phrase_words"] - 1, 1)

    for _ in range(passes):
        unigrams, bigrams, total = count_unigrams_and_bigrams(sequences)
        accepted_pairs: set[tuple[Token, Token]] = set()

        for (token_a, token_b), count in bigrams.items():
            words_a, words_b = _words_of(token_a), _words_of(token_b)
            if len(words_a) + len(words_b) > cfg["max_phrase_words"]:
                continue
            if count < cfg["min_pair_count"]:
                continue
            score = npmi(count, unigrams[token_a], unigrams[token_b], total)
            if score < cfg["min_npmi"]:
                continue

            accepted_pairs.add((token_a, token_b))
            words = words_a + words_b
            phrase = " ".join(words)
            all_phrases[phrase] = {
                "phrase": phrase,
                "words": list(words),
                "count": count,
                "npmi": round(score, 4),
            }

        if not accepted_pairs:
            break
        sequences = [merge_sequence(seq, accepted_pairs) for seq in sequences]

    pruned = prune_redundant_phrases(
        list(all_phrases.values()), redundancy_ratio=cfg["redundancy_ratio"]
    )
    return pruned

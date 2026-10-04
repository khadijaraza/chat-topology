"""Embed arbitrary-length text with a sentence-transformers model.

Shared by src/segmentation.py (window text) and src/grounded_extraction.py
(segment text): both need to embed spans of raw conversation text that can
easily exceed the embedding model's token limit (this corpus's turns are
heavy-tailed -- median 73 words, p90 522, max 3,588). Chunking + mean-
pooling rather than truncating avoids silently dropping the back half of
a long span.

get_embedding_model() lives here (not in src/legacy/concept_clustering.py,
where it originated) because both segmentation.py and grounded_extraction.py
-- current pipeline, not legacy -- need the same cached model loader; moving
it here let concept_clustering.py move to legacy wholesale instead of
staying a mixed current/legacy file. src/legacy/concept_clustering.py
imports it back from here.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str):
    """Load (and cache) the sentence-transformers model. Slow on first call
    per process -- downloads the model on first-ever use, then loads from
    the local cache."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def split_into_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split text into pieces each <= max_tokens, on sentence boundaries.

    Falls back to a word-level split for any single sentence that alone
    exceeds max_tokens (e.g. a code block or run-on paste with no
    punctuation).
    """
    if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        if sentence_tokens > max_tokens:
            flush()
            words = sentence.split()
            piece: list[str] = []
            piece_tokens = 0
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if piece_tokens + word_tokens > max_tokens and piece:
                    chunks.append(" ".join(piece))
                    piece, piece_tokens = [], 0
                piece.append(word)
                piece_tokens += word_tokens
            if piece:
                chunks.append(" ".join(piece))
            continue

        if current_tokens + sentence_tokens > max_tokens and current:
            flush()
        current.append(sentence)
        current_tokens += sentence_tokens

    flush()
    return chunks or [text]


def mean_pool_normalize(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def embed_long_text(text: str, model, max_tokens: int) -> np.ndarray:
    """Embed text, chunking + mean-pooling if it exceeds the model's token
    limit rather than silently truncating."""
    chunks = split_into_chunks(text, model.tokenizer, max_tokens)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    if len(chunks) == 1:
        return embeddings[0]
    return mean_pool_normalize(embeddings)

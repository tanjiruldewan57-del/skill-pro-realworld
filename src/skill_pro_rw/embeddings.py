"""Deterministic local text embeddings with optional sentence-transformers support."""

from __future__ import annotations

import math
import re
import zlib
from typing import Protocol
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class TextEmbedder(Protocol):
    """Protocol for text embedding providers returning L2-normalized vectors."""

    def embed(self, text: str) -> np.ndarray:
        """Encode text into a normalized 1D float32 vector."""


class LocalHashingNGramEmbedder:
    """Zero-dependency deterministic word + char-trigram hashing embedder.

    Uses zlib.crc32 (process-stable across restarts, unlike Python's built-in hash)
    to project word unigrams, word bigrams, and character trigrams into a fixed vector.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Encode input text into an L2-normalized float32 embedding."""
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = text.lower().strip()
        if not cleaned:
            return vec

        tokens = _TOKEN_RE.findall(cleaned)
        if not tokens:
            return vec

        # Word unigrams (weight 1.5) and word bigrams (weight 1.0)
        for tok in tokens:
            self._accumulate(vec, f"w:{tok}", 1.5)
        for idx in range(len(tokens) - 1):
            self._accumulate(vec, f"b:{tokens[idx]}_{tokens[idx + 1]}", 1.0)

        # Character trigrams (weight 0.5) for resilient morphological matching
        compact = " ".join(tokens)
        if len(compact) >= 3:
            for idx in range(len(compact) - 2):
                self._accumulate(vec, f"c:{compact[idx : idx + 3]}", 0.5)

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def _accumulate(self, vec: np.ndarray, feature: str, weight: float) -> None:
        digest = zlib.crc32(feature.encode("utf-8"))
        bucket = digest % self.dim
        sign = 1.0 if ((digest >> 16) & 1) == 0 else -1.0
        vec[bucket] += sign * weight


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute bounded cosine similarity between two embedding vectors."""
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim)) if math.isfinite(sim) else 0.0

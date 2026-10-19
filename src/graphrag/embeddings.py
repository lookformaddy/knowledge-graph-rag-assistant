"""Offline vector search via TF-IDF + cosine similarity.

Production deployments would plug a dense embedding model (e.g. a transformer
encoder) behind this same ``VectorStore`` interface. TF-IDF is used here so
the whole system — including the eval — runs deterministically offline with
zero model downloads, which makes failures debuggable and CI hermetic.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfVectorStore:
    """Minimal TF-IDF index over text documents with cosine search.

    Raw term frequencies are stored; IDF weighting is applied to a copy at
    search time, so documents can be added at any point without corrupting
    previously indexed vectors.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._tf: list[dict[str, float]] = []
        self._idf: dict[str, float] | None = None

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        n = max(len(tokens), 1)
        self._ids.append(doc_id)
        self._tf.append({t: c / n for t, c in counts.items()})
        self._idf = None  # invalidate; recomputed on next search

    def _ensure_idf(self) -> dict[str, float]:
        if self._idf is None:
            df: dict[str, int] = defaultdict(int)
            for vec in self._tf:
                for term in vec:
                    df[term] += 1
            total = max(len(self._tf), 1)
            self._idf = {t: math.log((1 + total) / (1 + c)) + 1.0 for t, c in df.items()}
        return self._idf

    @staticmethod
    def _norm(vec: dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) or 1.0

    def search(self, query: str, top_k: int = 4) -> list[tuple[str, float]]:
        """Return ``(doc_id, cosine_score)`` pairs, best first."""
        idf = self._ensure_idf()
        q_counts = Counter(tokenize(query))
        q_len = max(sum(q_counts.values()), 1)
        q_vec = {t: (c / q_len) * idf.get(t, 1.0) for t, c in q_counts.items()}
        q_norm = self._norm(q_vec)
        scored: list[tuple[str, float]] = []
        for doc_id, tf in zip(self._ids, self._tf):
            weighted = {t: v * idf.get(t, 1.0) for t, v in tf.items()}
            dot = sum(q_vec.get(t, 0.0) * v for t, v in weighted.items())
            scored.append((doc_id, dot / (q_norm * self._norm(weighted))))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [(d, round(s, 4)) for d, s in scored[:top_k] if s > 0.0]

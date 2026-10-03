"""Embedding-поиск и reranking по фрагментам документа."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .chunking import Chunk
from .clients import embed, rerank
from .config import Service


@dataclass
class Hit:
    chunk: Chunk
    embedding_score: float
    embedding_rank: int
    rerank_score: float | None = None
    rerank_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "page": self.chunk.page,
            "embedding_score": round(self.embedding_score, 6),
            "embedding_rank": self.embedding_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "rerank_rank": self.rerank_rank,
        }


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Index:
    """Простой in-memory индекс: вектор на фрагмент."""

    def __init__(self, service: Service, chunks: list[Chunk]):
        self.service = service
        self.chunks = chunks
        self.vectors = embed(service, [c.text for c in chunks])
        self.dim = len(self.vectors[0]) if self.vectors else 0

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        qvec = embed(self.service, [query])[0]
        scored = [(cosine(qvec, vec), chunk) for vec, chunk in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Hit(chunk=chunk, embedding_score=score, embedding_rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def apply_rerank(service: Service, query: str, hits: list[Hit], top_n: int = 3) -> list[Hit]:
    if not hits:
        return []
    ranked = rerank(service, query, [h.chunk.text for h in hits])
    out: list[Hit] = []
    for position, row in enumerate(ranked, start=1):
        hit = hits[row["index"]]
        hit.rerank_score = row["score"]
        hit.rerank_rank = position
        out.append(hit)
    return out[:top_n]

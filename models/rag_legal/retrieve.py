"""Hybrid BM25 + dense retrieval with RRF fusion and cross-encoder reranking."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np

from .corpus import load_chunks_jsonl

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:
    CrossEncoder = None  # type: ignore
    SentenceTransformer = None  # type: ignore


def _tokenize(s: str) -> list[str]:
    return s.lower().replace("\n", " ").split()


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def load_index_bundle(index_dir: str) -> dict[str, Any]:
    chunks = load_chunks_jsonl(os.path.join(index_dir, "chunks.jsonl"))
    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
        blob = pickle.load(f)
    bm25: BM25Okapi = blob["bm25"]
    emb_path = os.path.join(index_dir, "dense_embeddings.npy")
    emb = None
    if meta.get("has_dense") and os.path.isfile(emb_path):
        emb = np.load(emb_path)
    return {"chunks": chunks, "meta": meta, "bm25": bm25, "dense_embeddings": emb}


class HybridRetriever:
    def __init__(
        self,
        index_dir: str,
        *,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
        bi_encoder_model: str | None = None,
    ):
        if BM25Okapi is None:
            raise ImportError("pip install rank-bm25")
        bundle = load_index_bundle(index_dir)
        self.chunks = bundle["chunks"]
        self.meta = bundle["meta"]
        self.bm25 = bundle["bm25"]
        self.dense_embeddings = bundle["dense_embeddings"]
        self._id_to_idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.rrf_k = rrf_k

        self._bi_model = None
        if self.dense_embeddings is not None:
            name = bi_encoder_model or self.meta.get("dense_model_name") or "sentence-transformers/all-MiniLM-L6-v2"
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers required for dense retrieval")
            self._bi_model = SentenceTransformer(name)

        if CrossEncoder is None:
            raise ImportError("sentence-transformers required for cross-encoder reranking")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

    def _bm25_top(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:n]
        return [self.chunks[int(i)]["id"] for i in idx]

    def _dense_top(self, query: str, n: int) -> list[str]:
        if self.dense_embeddings is None or self._bi_model is None:
            return []
        q = self._bi_model.encode(query, normalize_embeddings=True)
        sim = self.dense_embeddings @ q.astype(np.float64)
        idx = np.argsort(-sim)[:n]
        return [self.chunks[int(i)]["id"] for i in idx]

    def retrieve(
        self,
        query: str,
        *,
        top_k_fuse: int = 40,
        top_k_rerank: int = 8,
        bm25_pool: int = 50,
        dense_pool: int = 50,
    ) -> list[dict[str, Any]]:
        lists: list[list[str]] = [self._bm25_top(query, bm25_pool)]
        d = self._dense_top(query, dense_pool)
        if d:
            lists.append(d)

        fused = reciprocal_rank_fusion(lists, k=self.rrf_k)[:top_k_fuse]
        cand_ids = [fid for fid, _ in fused]

        cand_ids_clean: list[str] = []
        texts: list[str] = []
        for cid in cand_ids:
            idx = self._id_to_idx.get(cid)
            if idx is None:
                continue
            cand_ids_clean.append(cid)
            texts.append(self.chunks[idx]["text"][:8000])

        if not texts:
            return []

        pairs = [(query, t) for t in texts]
        ce_scores = self.cross_encoder.predict(pairs)
        order = np.argsort(-np.array(ce_scores))[:top_k_rerank]

        out = []
        for j in order:
            cid = cand_ids_clean[j]
            idx = self._id_to_idx[cid]
            chunk = dict(self.chunks[idx])
            chunk["retriever_score"] = float(ce_scores[j])
            out.append(chunk)
        return out


def citation_overlap_score(answer: str, allowed_ids: set[str]) -> float:
    """Reward citations like [case_12] or [case_12_part1] that match retrieved chunk ids."""
    import re

    tags = re.findall(r"\[([a-zA-Z0-9_]+)\]", answer)
    if not tags:
        return 0.0
    hits = 0
    for t in tags:
        matched = False
        if t in allowed_ids:
            matched = True
        else:
            for aid in allowed_ids:
                if aid == t or aid.startswith(t + "_"):
                    matched = True
                    break
        if matched:
            hits += 1
    return hits / len(tags)

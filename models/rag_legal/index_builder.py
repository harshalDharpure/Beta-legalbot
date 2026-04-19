"""Build BM25 + dense embeddings index on disk."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np

from .corpus import load_corpus_file, save_chunks_jsonl

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    BM25Okapi = None  # type: ignore
    _BM25_ERR = e
else:
    _BM25_ERR = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


def _tokenize(s: str) -> list[str]:
    return s.lower().replace("\n", " ").split()


def build_index(
    corpus_path: str,
    out_dir: str,
    *,
    max_chunk_chars: int = 2500,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    skip_dense: bool = False,
) -> dict[str, Any]:
    if BM25Okapi is None:
        raise ImportError("Install rank_bm25: pip install rank-bm25") from _BM25_ERR

    os.makedirs(out_dir, exist_ok=True)
    chunks = load_corpus_file(corpus_path, max_chunk_chars=max_chunk_chars)
    chunks_path = os.path.join(out_dir, "chunks.jsonl")
    save_chunks_jsonl(chunks, chunks_path)

    texts = [c["text"] for c in chunks]
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)

    meta: dict[str, Any] = {
        "corpus_path": corpus_path,
        "n_chunks": len(chunks),
        "max_chunk_chars": max_chunk_chars,
        "dense_model_name": dense_model_name,
        "has_dense": False,
    }

    if not skip_dense:
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers for dense retrieval")
        st = SentenceTransformer(dense_model_name)
        emb = st.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        np.save(os.path.join(out_dir, "dense_embeddings.npy"), emb.astype(np.float32))
        meta["has_dense"] = True

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta

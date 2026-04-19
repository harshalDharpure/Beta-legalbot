#!/usr/bin/env python3
"""
Legal RAG CLI (run from repository root):

  python models/rag_legal/cli.py build-index \\
    --corpus experiments/exp2_pretraining_only/pretraining/legal_corpus/all_cases.txt \\
    --out-dir models/rag_legal_index

  python models/rag_legal/cli.py query \\
    --index-dir models/rag_legal_index \\
    --model llama3.1_8b \\
    --checkpoint models/llama3.1_8b/checkpoints/exp3/final \\
    --text "What is POCSO Section 19?"

  python models/rag_legal/cli.py eval-jsonl \\
    --index-dir models/rag_legal_index \\
    --model llama3.1_8b \\
    --checkpoint models/llama3.1_8b/checkpoints/exp3/final \\
    --test experiments/exp3_pretraining_finetuning/finetuning/test.jsonl \\
    --max-samples 20 \\
    --out-json models/rag_legal_results/exp3_rag_sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_MODELS = os.path.join(_REPO, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from tqdm import tqdm

from evaluate_generation import load_config, load_model_and_tokenizer
from evaluation.metrics import calculate_batch_metrics, calculate_nli_score

from models.rag_legal.index_builder import build_index


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_build_index(args: argparse.Namespace) -> None:
    meta = build_index(
        args.corpus,
        args.out_dir,
        max_chunk_chars=args.max_chunk_chars,
        dense_model_name=args.dense_model,
        skip_dense=args.skip_dense,
    )
    print(json.dumps(meta, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    from models.rag_legal.inference import build_rag_context, run_rag_best_of_n_for_prompt
    from models.rag_legal.retrieve import HybridRetriever

    retriever = HybridRetriever(
        args.index_dir,
        cross_encoder_model=args.cross_encoder,
    )
    cfg_path = os.path.join("models", args.model, "config.yaml")
    cfg = load_config(cfg_path) if os.path.isfile(cfg_path) else {}
    use_qlora = cfg.get("model", {}).get("use_qlora", False)
    model, tokenizer = load_model_and_tokenizer(args.model, args.checkpoint, use_qlora)
    try:
        user_block, retrieved, allowed_ids = build_rag_context(retriever, args.text)
        out = run_rag_best_of_n_for_prompt(
            model,
            tokenizer,
            retriever,
            model_name=args.model,
            user_block=user_block,
            allowed_ids=allowed_ids,
            user_query=args.text,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            ce_weight=args.ce_weight,
            cite_weight=args.cite_weight,
            temperature=args.temperature,
        )
    finally:
        del model, tokenizer
        import torch

        torch.cuda.empty_cache()

    slim = [
        {"id": x["id"], "case_id": x.get("case_id"), "preview": (x.get("text") or "")[:200]}
        for x in retrieved
    ]
    print(json.dumps({**out, "retrieved": slim}, indent=2, ensure_ascii=False))


def cmd_eval_jsonl(args: argparse.Namespace) -> None:
    from models.rag_legal.inference import build_rag_context, run_rag_best_of_n_for_prompt
    from models.rag_legal.retrieve import HybridRetriever

    retriever = HybridRetriever(
        args.index_dir,
        cross_encoder_model=args.cross_encoder,
    )
    cfg_path = os.path.join("models", args.model, "config.yaml")
    cfg = load_config(cfg_path) if os.path.isfile(cfg_path) else {}
    use_qlora = cfg.get("model", {}).get("use_qlora", False)
    model, tokenizer = load_model_and_tokenizer(args.model, args.checkpoint, use_qlora)

    data = _load_jsonl(args.test)
    if args.max_samples:
        data = data[: args.max_samples]

    refs = []
    cands = []
    details = []
    try:
        for row in tqdm(data, desc="RAG eval"):
            q = row.get("input", "").strip()
            ref = row.get("output", "").strip()
            if not q:
                continue
            user_block, retrieved, allowed_ids = build_rag_context(retriever, q)
            out = run_rag_best_of_n_for_prompt(
                model,
                tokenizer,
                retriever,
                model_name=args.model,
                user_block=user_block,
                allowed_ids=allowed_ids,
                user_query=q,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                ce_weight=args.ce_weight,
                cite_weight=args.cite_weight,
                temperature=args.temperature,
            )
            cand = out["answer"]
            refs.append(ref)
            cands.append(cand)
            details.append(
                {
                    "dialogue_id": row.get("dialogue_id"),
                    "best_index": out["best_index"],
                    "retrieved_ids": [r["id"] for r in retrieved],
                    "candidates": out["candidates"],
                }
            )
    finally:
        del model, tokenizer
        import torch

        torch.cuda.empty_cache()

    metrics = calculate_batch_metrics(refs, cands, lang="en")
    try:
        metrics.update(calculate_nli_score(refs, cands))
    except Exception as e:
        metrics["nli_error"] = str(e)

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"metrics": metrics, "n": len(refs), "per_sample": details}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Legal hybrid RAG + best-of-N pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="Chunk corpus, BM25 + dense embeddings")
    b.add_argument("--corpus", required=True, help="Path to all_cases.txt")
    b.add_argument("--out-dir", required=True, help="Output index directory")
    b.add_argument("--max-chunk-chars", type=int, default=2500)
    b.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--skip-dense", action="store_true", help="BM25 + reranker only")
    b.set_defaults(func=cmd_build_index)

    q = sub.add_parser("query", help="Single query with RAG + best-of-N")
    q.add_argument("--index-dir", required=True)
    q.add_argument("--model", required=True, help="e.g. llama3.1_8b")
    q.add_argument("--checkpoint", required=True, help="Path to SFT checkpoint (final/)")
    q.add_argument("--text", required=True)
    q.add_argument("--n-samples", type=int, default=4)
    q.add_argument("--max-new-tokens", type=int, default=384)
    q.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    q.add_argument("--ce-weight", type=float, default=1.0)
    q.add_argument("--cite-weight", type=float, default=0.35)
    q.add_argument("--temperature", type=float, default=0.72)
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval-jsonl", help="Evaluate on test.jsonl (input/output)")
    e.add_argument("--index-dir", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--test", required=True, help="JSONL with input + output")
    e.add_argument("--max-samples", type=int, default=0, help="0 = all")
    e.add_argument("--out-json", required=True)
    e.add_argument("--n-samples", type=int, default=4)
    e.add_argument("--max-new-tokens", type=int, default=384)
    e.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    e.add_argument("--ce-weight", type=float, default=1.0)
    e.add_argument("--cite-weight", type=float, default=0.35)
    e.add_argument("--temperature", type=float, default=0.72)
    e.set_defaults(func=cmd_eval_jsonl)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

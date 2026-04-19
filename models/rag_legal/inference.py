"""Load HF checkpoint, run RAG prompt, best-of-N selection."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_MODELS = os.path.join(_REPO, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from evaluate_generation import (  # noqa: E402
    format_prompt,
    load_config,
    load_model_and_tokenizer,
)
from .prompts import build_rag_user_block
from .retrieve import HybridRetriever, citation_overlap_score


def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 384,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model_name: str | None = None,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens=1,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    gen = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return gen.strip()


def build_rag_context(retriever: HybridRetriever, user_query: str) -> tuple[str, list[dict[str, Any]], set[str]]:
    retrieved = retriever.retrieve(user_query)
    allowed_ids = {c["id"] for c in retrieved}
    user_block = build_rag_user_block(user_query, retrieved)
    return user_block, retrieved, allowed_ids


def pick_best_candidate(
    retriever: HybridRetriever,
    user_query: str,
    candidates: list[str],
    allowed_ids: set[str],
    *,
    ce_weight: float = 1.0,
    cite_weight: float = 0.35,
) -> tuple[str, int, list[dict[str, Any]]]:
    pairs = [(user_query, c) for c in candidates]
    ce_scores = retriever.cross_encoder.predict(pairs)
    best_i = 0
    best_score = -1e9
    scored = []
    for i, c in enumerate(candidates):
        cite = citation_overlap_score(c, allowed_ids)
        total = float(ce_scores[i]) * ce_weight + cite * cite_weight
        scored.append(
            {"text": c, "cross_encoder": float(ce_scores[i]), "citation_score": cite, "combined": total}
        )
        if total > best_score:
            best_score = total
            best_i = i
    return candidates[best_i], best_i, scored


def run_rag_best_of_n_for_prompt(
    model,
    tokenizer,
    retriever: HybridRetriever,
    *,
    model_name: str,
    user_block: str,
    allowed_ids: set[str],
    user_query: str,
    n_samples: int = 4,
    max_new_tokens: int = 384,
    ce_weight: float = 1.0,
    cite_weight: float = 0.35,
    temperature: float = 0.72,
) -> dict[str, Any]:
    prompt = format_prompt(user_block, model_name=model_name)
    candidates = [
        generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            model_name=model_name,
        )
    ]
    for _ in range(max(0, n_samples - 1)):
        candidates.append(
            generate_one(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                model_name=model_name,
            )
        )
    answer, best_i, scored = pick_best_candidate(
        retriever, user_query, candidates, allowed_ids, ce_weight=ce_weight, cite_weight=cite_weight
    )
    return {
        "answer": answer,
        "best_index": best_i,
        "candidates": scored,
    }


def run_rag_best_of_n(
    *,
    model_name: str,
    checkpoint_path: str,
    retriever: HybridRetriever,
    user_query: str,
    n_samples: int = 4,
    max_new_tokens: int = 384,
    ce_weight: float = 1.0,
    cite_weight: float = 0.35,
    temperature: float = 0.72,
) -> dict[str, Any]:
    user_block, retrieved, allowed_ids = build_rag_context(retriever, user_query)

    config_path = os.path.join("models", model_name, "config.yaml")
    full_cfg = os.path.join(_REPO, config_path)
    cfg = load_config(config_path) if os.path.isfile(full_cfg) else {}
    use_qlora = cfg.get("model", {}).get("use_qlora", False)

    model, tokenizer = load_model_and_tokenizer(model_name, checkpoint_path, use_qlora)
    try:
        out = run_rag_best_of_n_for_prompt(
            model,
            tokenizer,
            retriever,
            model_name=model_name,
            user_block=user_block,
            allowed_ids=allowed_ids,
            user_query=user_query,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            ce_weight=ce_weight,
            cite_weight=cite_weight,
            temperature=temperature,
        )
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()

    out["retrieved"] = [
        {"id": x["id"], "case_id": x.get("case_id"), "preview": x["text"][:240]} for x in retrieved
    ]
    return out

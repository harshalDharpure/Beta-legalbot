# RAG pipeline — results (tables)

**Standalone from Exp1–Exp5 category tables.** Raw metrics JSON: `models/rag_legal_results/exp3_rag_eval_40.json` (when generated via `cli.py eval-jsonl`).

---

## Table 1 — Pipeline overview

| Layer | What it does | Main code |
|-------|----------------|-----------|
| Index | Chunk corpus, BM25 + dense embeddings, `meta.json` | `models/rag_legal/index_builder.py`, `cli.py build-index` |
| Retrieve | Hybrid search + RRF + cross-encoder rerank | `models/rag_legal/retrieve.py` |
| Generate | Citation-grounded prompt, best-of-N scoring | `models/rag_legal/inference.py`, `cli.py query` / `eval-jsonl` |

---

## Table 2 — Corpus, index, and eval data

| Item | Path / value |
|------|----------------|
| Legal corpus (chunking source) | `experiments/exp2_pretraining_only/pretraining/legal_corpus/all_cases.txt` |
| On-disk index | `models/rag_legal_index/` (`chunks.jsonl`, `bm25.pkl`, `dense_embeddings.npy`, `meta.json`) |
| RAG eval test set | `experiments/exp3_pretraining_finetuning/finetuning/test.jsonl` |
| RAG eval raw JSON | `models/rag_legal_results/exp3_rag_eval_40.json` |
| Eval sample size | **40** |

---

## Table 3 — RAG evaluation metrics (generator = Llama 3.1 8B Exp3 SFT)

Setting: hybrid retrieval + rerank + citation prompt + best-of-N; checkpoint `models/llama3.1_8b/checkpoints/exp3/final`.

| Metric | Value |
|--------|-------|
| ROUGE-1 F1 | 0.0978 |
| ROUGE-2 F1 | 0.0143 |
| ROUGE-L F1 | 0.0590 |
| BLEU-1 | 0.0481 |
| BLEU-2 | 0.0169 |
| BLEU-3 | 0.0076 |
| BLEU-4 | 0.0036 |
| METEOR | 0.0667 |
| NLI score | 0.1323 |
| BERTScore F1 | 0.0000 (not computed in this run) |

---

*Last updated: 2026-04-03. Table 3 matches the RAG eval run recorded in `exp3_rag_eval_40.json`.*

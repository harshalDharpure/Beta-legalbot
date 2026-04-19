# Legal RAG pipeline (retrieval + grounded generation)

This document explains **what we built**, **how it works**, and **how to run it** for a supervisor or paper (e.g. EMNLP, NAACL). It covers **hybrid retrieval**, **citation-style prompting**, and **best-of-N** selection only—**not** preference optimization (DPO) or reward models.

---

## 1. What problem does this solve?

Fine-tuned LLMs (your Exp1–Exp3 setup) answer user questions **from memory**. In law, that can lead to:

- **Hallucinated statutes or case facts** (plausible but wrong).
- **No traceability**: the user cannot verify where an answer came from.
- **Weak grounding** when the question needs **specific judgment language** from your domain corpus.

This pipeline adds **retrieval** (evidence from real legal text), **structured prompting** (answer with citations), and **selection** among multiple candidates (best-of-N).

---

## 2. High-level architecture

```
                    ┌─────────────────────────────────────┐
  User question ──► │  Hybrid retrieval (BM25 + dense)    │
                    │  → RRF fusion → Cross-encoder rerank │
                    └──────────────┬──────────────────────┘
                                   │ top-k passages + IDs
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Citation-grounded prompt            │
                    │  (passages labeled [case_12], …)      │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Fine-tuned LLM (your checkpoint)    │
                    │  generates N candidates              │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Best-of-N scoring                   │
                    │  CE(query, answer) + citation bonus  │
                    └──────────────┬──────────────────────┘
                                   ▼
                              Final answer
```

---

## 3. What each component does

### 3.1 Corpus and chunking

- **Input:** Plain legal text (e.g. `all_cases.txt`) with markers like `[case N]`.
- **Process:** Text is split into **chunks** with stable IDs (`case_5`, `case_5_part1`, …) so every retrieved span can be **cited** in the answer.
- **Where:** `models/rag_legal/corpus.py`, built via `cli.py build-index`.

### 3.2 Hybrid retrieval (BM25 + dense)

- **BM25 (lexical):** Strong for exact legal terms, section numbers, names.
- **Dense embeddings (bi-encoder):** Strong for paraphrases and semantic similarity.
- **RRF (Reciprocal Rank Fusion):** Combines both ranked lists into one fused ranking.

### 3.3 Cross-encoder reranking

- A **cross-encoder** scores **(query, passage)** pairs jointly.
- Candidates after fusion are **reranked** so the generator sees the most relevant passages first.
- **Default model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### 3.4 Citation-grounded generation

- The prompt includes **PASSAGES** with explicit bracket IDs, e.g. `[case_3]`, and instructs the model to ground factual claims and cite IDs when appropriate.
- **Format** matches your training: `User: …` / `Assistant:` (`models/evaluate_generation.py` / training template).

### 3.5 Best-of-N selection

- The LLM produces multiple completions (e.g. one greedy + several sampled).
- Each candidate is scored with a **cross-encoder** on (question, answer) plus a **citation overlap** bonus vs retrieved chunk IDs.
- The **highest combined score** is returned.

---

## 4. File map

| Piece | Location |
|-------|----------|
| Chunking & index build | `models/rag_legal/corpus.py`, `models/rag_legal/index_builder.py` |
| Retrieval + rerank + citation score | `models/rag_legal/retrieve.py` |
| Prompts | `models/rag_legal/prompts.py` |
| Generation + best-of-N | `models/rag_legal/inference.py` |
| CLI (`build-index`, `query`, `eval-jsonl`) | `models/rag_legal/cli.py` |
| On-disk index (after build) | `models/rag_legal_index/` |
| RAG eval results | `models/rag_legal_results/` (e.g. `exp3_rag_eval_40.json`) |
| Results tables (markdown) | `docs/markdown/models/evaluation_results/rag_pipeline_results.md` |
| Dependencies | `models/requirements.txt` (`rank-bm25`, `sentence-transformers`) |

---

## 5. Commands (from repository root)

**Build the index** (BM25 + dense embeddings over `all_cases.txt`):

```bash
python models/rag_legal/cli.py build-index \
  --corpus experiments/exp2_pretraining_only/pretraining/legal_corpus/all_cases.txt \
  --out-dir models/rag_legal_index
```

**Single query:**

```bash
python models/rag_legal/cli.py query \
  --index-dir models/rag_legal_index \
  --model llama3.1_8b \
  --checkpoint models/llama3.1_8b/checkpoints/exp3/final \
  --text "Your question here"
```

**Evaluate on a JSONL test set** (`input` / `output` fields, same as Exp3):

```bash
python models/rag_legal/cli.py eval-jsonl \
  --index-dir models/rag_legal_index \
  --model llama3.1_8b \
  --checkpoint models/llama3.1_8b/checkpoints/exp3/final \
  --test experiments/exp3_pretraining_finetuning/finetuning/test.jsonl \
  --max-samples 40 \
  --out-json models/rag_legal_results/exp3_rag_eval_40.json
```

---

## 6. Limitations (honest reporting)

- **Retrieval can fail:** Wrong or empty top-k → the model may still speculate; mitigations include abstention prompts and retrieval-score thresholds.
- **Citations can be wrong:** Models may cite IDs incorrectly; mitigations include stricter decoding or human review.
- **Metrics are automatic proxies:** **Legal correctness** for a top-tier claim often needs **expert** or structured human evaluation.

---

## 7. Relation to Exp1–Exp5

Your pretraining, fine-tuning, zero-shot, and few-shot experiments train **the generator**. This pipeline **wraps** a chosen checkpoint (e.g. Exp3) with **retrieval, reranking, and best-of-N** at inference time. Present it as a separate **system** row or section: e.g. “Grounded generation with hybrid RAG.”

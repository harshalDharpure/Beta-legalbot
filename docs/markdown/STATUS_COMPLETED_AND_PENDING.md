# Project Status: Completed vs Not Completed

**Last updated:** February 2026

---

# ✅ COMPLETED

## 1. Dataset & setup
| Item | Details |
|------|---------|
| 70/10/20 split | `experiments/exp1_finetuning_only/data/` — train_70.jsonl, val_10.jsonl, test_20.jsonl |
| Generation format | Dialogue data with `input` / `output` for Exp1 |
| Model configs | All 7 models: configs in `models/{model}/config.yaml` |
| QLoRA / full fine-tuning | Set up for generation (QLoRA) and encoder models |
| Evaluation code | BLEU, ROUGE, METEOR, BERTScore in `evaluation/metrics.py` |

## 2. Exp1 training (fine-tuning only)
| Model | Status | Checkpoint path |
|-------|--------|-----------------|
| LLaMA-3.1-8B | ✅ Done | `models/llama3.1_8b/checkpoints/exp1/final` |
| Mistral-7B | ✅ Done | `models/mistral_7b/checkpoints/exp1/final` |
| Qwen2.5-7B | ✅ Done | `models/qwen2.5_7b/checkpoints/exp1/final` |
| Qwen2.5-1.5B | ✅ Done | `models/qwen2.5_1.5b/checkpoints/exp1/final` |
| Phi-3-mini | ✅ Done | `models/phi3_mini/checkpoints/exp1/final` |

## 3. Legal corpus for pretraining
| Item | Details |
|------|---------|
| Corpus file | `code_mixed_posco_dataset/all_cases.txt` copied to Exp2/Exp3 legal corpus dirs |
| Corpus location | `experiments/exp2_pretraining_only/pretraining/legal_corpus/all_cases.txt` |
| Setup script | `python data/prepare_legal_corpus.py --use-all-cases` |

## 4. Exp1 evaluation
| Model | Has exp1_results.json |
|-------|------------------------|
| LLaMA-3.1-8B | ✅ Yes |
| Mistral-7B | ✅ Yes |
| Qwen2.5-7B | ✅ Yes |
| Qwen2.5-1.5B | ✅ Yes |
| Phi-3-mini | ✅ Yes |

## 5. Model comparison tables
| Item | Details |
|------|---------|
| Table generator | `models/generate_model_comparison_tables.py` |
| Output | `models/evaluation_results/model_comparison_tables_*.md` |
| Tables 1, 4, 5 | Generated (overall metrics, length, ranking) |
| Tables 2, 3 | Need re-run of evaluation (language/complexity breakdown) |

## 6. Encoder models (separate pipeline)
| Item | Status |
|------|--------|
| XLM-RoBERTa-Large | Trained and evaluated (see `models/EVALUATION_RESULTS.md`) |
| MuRIL-Large | Trained and evaluated |

## 7. Scripts and automation
| Script | Purpose |
|--------|---------|
| `START_EXP1_EVALUATION.sh` | Run Exp1 evaluation for all 5 models (one GPU, smallest first) |
| `START_EXP2_EVALUATION.sh` | Run Exp2 evaluation on multiple GPUs (set `GPUS="1 4"` etc.) |
| `START_EXP3_TRAINING.sh` | Exp3: finetune all 5 from Exp2 on dialogue (use free GPU: `CUDA_VISIBLE_DEVICES=0 ./START_EXP3_TRAINING.sh`) |
| `models/evaluate_generation.py` | Single-model evaluation; saves metrics + optional language/complexity breakdown |
| `models/evaluate_all_exp1.py` | Batch Exp1 evaluation |

## 8. Exp2 evaluation results (zero-shot, 968 test samples)
Pretraining-only models evaluated on `experiments/exp2_pretraining_only/evaluation/test.jsonl`. Results in `models/{model}/results/exp2_results.json`.

| Model | BLEU-1 | ROUGE-1 F1 | ROUGE-L F1 | METEOR |
|-------|--------|------------|------------|--------|
| LLaMA-3.1-8B | 0.154 | **0.219** | **0.159** | **0.151** |
| Qwen2.5-7B | 0.126 | **0.217** | 0.142 | 0.142 |
| Mistral-7B | 0.090 | 0.164 | 0.096 | 0.107 |
| Phi-3-mini | 0.092 | 0.140 | 0.084 | 0.104 |
| Qwen2.5-1.5B | 0.058 | 0.125 | 0.065 | 0.086 |

**Summary:** LLaMA-3.1-8B and Qwen2.5-7B lead on ROUGE-1/ROUGE-L/METEOR; Qwen2.5-1.5B is lowest (smallest model). All models do best on English and professional complexity in the per-language/per-complexity breakdowns inside each `exp2_results.json`.

---

# ❌ NOT COMPLETED

## 1. Exp1 evaluation (optional)
- [x] Phi-3-mini has `exp1_results.json`.
- [ ] (Optional) Re-run all 5 to get language/complexity breakdown for Tables 2 & 3

## 2. Exp2 pretraining
- [x] Pretraining script: `models/pretrain_template.py` (4-bit QLoRA or full + bf16, chunked legal corpus)
- [x] Runner: `./START_EXP2_PRETRAINING.sh` (all 5 models); `./RUN_EXP2_QWEN_1.5B_ONLY.sh` (Qwen2.5-1.5B only)
- [x] **5/5 models** have Exp2 checkpoints → `models/{model}/checkpoints/exp2/pretrained/final` (LLaMA-3.1-8B, Mistral-7B, Qwen2.5-7B, Phi-3-mini, **Qwen2.5-1.5B**)
- [x] **Exp2 evaluation done** — all 5 models: `models/{model}/results/exp2_results.json` (run via `GPUS="1 4" ./START_EXP2_EVALUATION.sh`)

## 3. Exp3 full pipeline
- [x] **Pipeline ready:** `train_generation_template.py` loads from Exp2 and finetunes on `experiments/exp3_pretraining_finetuning/finetuning/`; `START_EXP3_TRAINING.sh` runs all 5 models with `--gpu 0`.
- [ ] **Run when a GPU is free:** `CUDA_VISIBLE_DEVICES=0 ./START_EXP3_TRAINING.sh` (each model needs ~30GB+ free GPU). Then evaluate: `python3 models/evaluate_generation.py --model <model> --experiment exp3`.
- [ ] Save to `models/{model}/checkpoints/exp3/final` and run Exp3 evaluation for all 5.

## 4. Exp4 zero-shot transfer
- [ ] Create cross-lingual splits
- [ ] Train and evaluate

## 5. Exp5 few-shot learning
- [ ] Create few-shot splits
- [ ] Train and evaluate

## 6. Phase 8 – evaluation & analysis
- [x] **Comparison tables generated:** `python3 models/generate_model_comparison_tables.py` → `models/evaluation_results/model_comparison_tables_*.md` (Table 1–5 from Exp1 results).
- [ ] Comprehensive comparison across Exp1/Exp2/Exp3 once Exp3 is done
- [ ] Ablation study (if planned)

---

# Quick reference

| Phase | Completed | Not completed |
|-------|-----------|----------------|
| Dataset & setup | ✅ Split, format, configs, metrics | — |
| Exp1 training | ✅ 5/5 models | — |
| Exp1 evaluation | ✅ 5/5 have results | Optional: Tables 2 & 3 re-run |
| Legal corpus | ✅ all_cases.txt in place | — |
| Exp2 pretraining | ✅ 5/5 done | — |
| Exp2 evaluation | ✅ 5/5 results | — |
| Exp3 | Script + data ready | Run `START_EXP3_TRAINING.sh` when GPU free |
| Exp4 | — | Zero-shot transfer |
| Exp5 | — | Few-shot learning |
| Phase 8 | ✅ Tables generated | Full cross-exp comparison after Exp3 |

---

*To regenerate this view, update this file after completing tasks.*

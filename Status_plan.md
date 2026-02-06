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

## 4. Exp1 evaluation (partial)
| Model | Has exp1_results.json |
|-------|------------------------|
| LLaMA-3.1-8B | ✅ Yes |
| Mistral-7B | ✅ Yes |
| Qwen2.5-7B | ✅ Yes |
| Qwen2.5-1.5B | ✅ Yes |
| Phi-3-mini | ❌ No (pending) |

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
| `models/evaluate_generation.py` | Single-model evaluation; saves metrics + optional language/complexity breakdown |
| `models/evaluate_all_exp1.py` | Batch Exp1 evaluation |

---

# ❌ NOT COMPLETED

## 1. Exp1 evaluation (remaining)
- [ ] Phi-3-mini: run evaluation and save `exp1_results.json`
- [ ] (Optional) Re-run all 5 to get language/complexity breakdown for Tables 2 & 3

## 2. Exp2 pretraining
- [ ] Implement pretraining script (e.g. `pretrain.py` or use `models/pretrain_template.py`)
- [ ] Pretrain all 5 generation models on legal corpus (`all_cases.txt`)
- [ ] Save checkpoints to `models/{model}/checkpoints/exp2/pretrained`
- [ ] Evaluate Exp2 models (zero-shot on test set)

## 3. Exp3 full pipeline
- [ ] Finetune all 5 models from Exp2 checkpoints on dialogue data
- [ ] Save to `models/{model}/checkpoints/exp3/final`
- [ ] Evaluate on test set

## 4. Exp4 zero-shot transfer
- [ ] Create cross-lingual splits
- [ ] Train and evaluate

## 5. Exp5 few-shot learning
- [ ] Create few-shot splits
- [ ] Train and evaluate

## 6. Phase 8 – evaluation & analysis
- [ ] Comprehensive comparison across all experiments
- [ ] Final paper tables and figures
- [ ] Ablation study (if planned)

---

# Quick reference

| Phase | Completed | Not completed |
|-------|-----------|----------------|
| Dataset & setup | ✅ Split, format, configs, metrics | — |
| Exp1 training | ✅ 5/5 models | — |
| Exp1 evaluation | ✅ 4/5 have results | Phi-3-mini; full Tables 2 & 3 need re-run |
| Legal corpus | ✅ all_cases.txt in place | — |
| Exp2 pretraining | — | Script, pretrain 5 models, evaluate |
| Exp3 | — | Full pipeline |
| Exp4 | — | Zero-shot transfer |
| Exp5 | — | Few-shot learning |
| Phase 8 | — | Final evaluation & paper |

---

*To regenerate this view, update this file after completing tasks.*

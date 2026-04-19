# Q1-style 3-stage legal dialogue research pipeline

Reproducible **Stage 1 (SFT)** → **Stage 2 (multi-objective)** → **Stage 3 (DPO)** with strict **train / validation / test** protocol.

## Directory layout

| Path | Role |
|------|------|
| `configs/` | Default YAML (`pipeline_default.yaml`) |
| `data/` | `prepare_splits.py`, `merge_train_val.py`, `data/splits/*.jsonl` (generated) |
| `stage1_sft/` | Masked causal LM SFT (`train.py`) |
| `stage2_multi_objective/` | \(L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}\) (`train.py`, `losses.py`) |
| `stage3_dpo/` | TRL DPO (`train.py`) |
| `evaluation/` | Metric aggregation, `stats.py` (bootstrap / paired t scaffolding) |
| `ablation/` | `run_stage2_ablations.py` |
| `checkpoints/` | Saved runs (create on first train) |
| `logs/` | Optional logs |

Run all commands from the **repository root** with:

```bash
export PYTHONPATH=/path/to/legal-bot
```

## Data protocol (critical)

1. **Split** once: `train`, `val`, `test` — use `data/prepare_splits.py`.
2. **Development:** train on `train` only; tune λ, β, early stopping on **val** only.
3. **Never** use `test` until the final paper run.
4. **Final retrain (optional):** `merge_train_val.py` → `final_train.jsonl`; retrain M2/M3 with **frozen** hyperparameters; evaluate **once** on `test`.

## Quickstart

### 1) Splits

```bash
python research_pipeline/data/prepare_splits.py \
  --source experiments/exp3_pretraining_finetuning/finetuning/train.jsonl \
  --out-dir research_pipeline/data/splits \
  --ratios 0.8 0.1 0.1 \
  --seed 42
```

### 2) Stage 1 — SFT (M1)

```bash
python research_pipeline/stage1_sft/train.py \
  --config research_pipeline/configs/pipeline_default.yaml \
  --train-jsonl research_pipeline/data/splits/train.jsonl \
  --val-jsonl research_pipeline/data/splits/val.jsonl \
  --output-dir research_pipeline/checkpoints/stage1/M1_s42
```

### 3) Stage 2 — Multi-objective (M2)

Ablations: `gen_only`, `gen_entail`, `gen_triplet`, `full`.

```bash
python research_pipeline/stage2_multi_objective/train.py \
  --config research_pipeline/configs/pipeline_default.yaml \
  --init-from base \
  --ablation full \
  --train-jsonl research_pipeline/data/splits/train.jsonl \
  --val-jsonl research_pipeline/data/splits/val.jsonl \
  --output-dir research_pipeline/checkpoints/stage2/M2_base_full_s42 \
  --seed 42
```

`--init-from exp3` loads **merged** weights from `project.exp3_checkpoint` in YAML (adds a **new** LoRA for Stage 2).

**L_entail (implemented):** frozen sentence encoder; minimize \(1 - \cos(\text{pooled LM hidden}, \text{embed}(y_{ref}))\) — differentiable semantic alignment.  
**Strict DeBERTa-KL** with discrete decode does not backprop to the LM; use as **monitoring** or extend with STE/REINFORCE if you add that experiment.

**L_triplet:** `triplet_proj(LM pooled) vs ST(y+) vs ST(y-)` with margin loss — negatives from `negative_output` field or in-batch swap (`hard_negatives.py`).

### 4) Stage 3 — DPO (M3)

```bash
python research_pipeline/stage3_dpo/train.py \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-path research_pipeline/checkpoints/stage2/M2_base_full_s42/adapter \
  --preferences data/preference_pairs.jsonl \
  --output-dir research_pipeline/checkpoints/stage3/M3_beta0.1 \
  --beta 0.1
```

`preference_pairs.jsonl`: `{"prompt":"User: ...\\nAssistant:","chosen":" ...","rejected":" ..."}`

### 5) Ablations (Stage 2)

```bash
python research_pipeline/ablation/run_stage2_ablations.py \
  --init-from base \
  --out-root research_pipeline/checkpoints/stage2_ablations
```

### 6) Evaluation

Use existing `models/evaluate_generation.py` on the **test** JSONL for full model metrics, or `research_pipeline/evaluation/run_eval.py` if you have reference/candidate pairs.

## Multi-seed / statistics

Re-run training with `--seed` ∈ {42, 43, 44}; aggregate metrics; use `evaluation/stats.py` for bootstrap CI and paired t scaffolding (add `scipy` for p-values if desired).

## Dependencies

See `requirements-research.txt` (in addition to project `models/requirements.txt`).

## Research claim (draft)

Staged training with **semantic alignment** (entailment-style cosine) and **contrastive triplet** terms improves **faithfulness** proxies before **preference alignment (DPO)**; report ablations and human legal rubric for Q1 rigor.

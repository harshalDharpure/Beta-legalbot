# Plan: Separate Files for New Models

This document lists **exactly which separate files** will be created for each **new** model. We do **not** add any model that is already used in your experiments.

---

## Models already in your experiments (do NOT add)

These 5 models are already used across training and evaluation; **no new files** are added for them:

| Key | Model | Hugging Face / Source |
|-----|--------|------------------------|
| `llama3.1_8b` | LLaMA 3.1 8B Instruct | meta-llama/Meta-Llama-3.1-8B-Instruct |
| `mistral_7b` | Mistral 7B Instruct v0.3 | mistralai/Mistral-7B-Instruct-v0.3 |
| `qwen2.5_7b` | Qwen 2.5 7B Instruct | Qwen/Qwen2.5-7B-Instruct |
| `qwen2.5_1.5b` | Qwen 2.5 1.5B Instruct | Qwen/Qwen2.5-1.5B-Instruct |
| `phi3_mini` | Phi-3 mini 4k Instruct | microsoft/Phi-3-mini-4k-instruct |

Only **new** models (latest families, not in the table above) get configs and READMEs below.

---

## New models only (latest; not already used)

- **Gemma 3** — You do not use any Gemma today; these are the latest (4B, 12B, 27B).
- **Qwen 3 8B** — Newer than Qwen 2.5; different from your existing `qwen2.5_7b` and `qwen2.5_1.5b`.
- **Llama 4 Scout** — Newer than LLaMA 3.1; different from your existing `llama3.1_8b`.
- **Gemini** (optional) — API-only; not a local model.

---

## Current Setup (Reference)

Each existing model (e.g. `qwen2.5_7b`) has:

| File / Folder        | Purpose |
|----------------------|--------|
| `models/<name>/config.yaml` | Model ID, tokenizer, QLoRA/quantization, training/data paths |
| `models/<name>/README.md`   | Short doc: model info, how to train/evaluate |
| `models/<name>/checkpoints/`| Created at runtime when training |
| `models/<name>/logs/`       | Created at runtime when training |
| `models/<name>/results/`    | Created at runtime when evaluating |

Training and evaluation use **shared** scripts:

- **Train:** `models/train_generation_template.py --model <name> --experiment exp1` (etc.)
- **Eval:** `models/evaluate_generation.py --model <name> --experiment exp1`

So for each new model we only add **two files per model**: `config.yaml` and `README.md`.

---

## New Models and Their Separate Files

### 1. Gemma 3 – 4B (recommended: 1× 40GB GPU)

| File | Path |
|------|------|
| Config | `models/gemma3_4b/config.yaml` |
| Readme | `models/gemma3_4b/README.md` |

- **Hugging Face:** `google/gemma-3-4b-it`
- **QLoRA:** Yes, 4-bit (fits easily on 40GB)

---

### 2. Gemma 3 – 12B (recommended: 1× 40GB GPU)

| File | Path |
|------|------|
| Config | `models/gemma3_12b/config.yaml` |
| Readme | `models/gemma3_12b/README.md` |

- **Hugging Face:** `google/gemma-3-12b-it`
- **QLoRA:** Yes, 4-bit

---

### 3. Gemma 3 – 27B (recommended: 1× 40GB GPU with 4-bit)

| File | Path |
|------|------|
| Config | `models/gemma3_27b/config.yaml` |
| Readme | `models/gemma3_27b/README.md` |

- **Hugging Face:** `google/gemma-3-27b-it`
- **QLoRA:** Yes, 4-bit (requires `bitsandbytes`; already supported in `train_generation_template.py`)

---

### 4. Qwen 3 – 8B (recommended: 1× 40GB GPU)

| File | Path |
|------|------|
| Config | `models/qwen3_8b/config.yaml` |
| Readme | `models/qwen3_8b/README.md` |

- **Hugging Face:** `Qwen/Qwen3-8B-Instruct` (latest Qwen3; not Qwen2.5 which you already use)
- **QLoRA:** Yes, 4-bit (or none for 8B on 40GB)

---

### 5. Llama 4 Scout – 17B active (recommended: 1× 40GB GPU with 4-bit)

| File | Path |
|------|------|
| Config | `models/llama4_scout/config.yaml` |
| Readme | `models/llama4_scout/README.md` |

- **Hugging Face:** `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- **QLoRA:** Yes, 4-bit (MoE; 17B active params)
- **Note:** Latest Llama 4 family (you already use LLaMA 3.1). Requires Meta’s Llama 4 license and Hugging Face access approval.

---

### 6. Gemini (optional – API only, no local GPU)

| File | Path |
|------|------|
| Config | `models/gemini_api/config.yaml` (optional: API key, model variant) |
| Readme | `models/gemini_api/README.md` |

- **Type:** Google API (e.g. `gemini-2.0-flash`). Not run on your GPUs; separate evaluation path (e.g. call API and compare to same test set).
- **Recommendation:** Treat as optional “API baseline” and add only if you want to compare against Gemini; no training, only evaluation script changes.

---

## Summary: Files to Create (per model)

| Model           | config.yaml              | README.md                |
|-----------------|--------------------------|--------------------------|
| gemma3_4b       | `models/gemma3_4b/config.yaml`       | `models/gemma3_4b/README.md`       |
| gemma3_12b      | `models/gemma3_12b/config.yaml`      | `models/gemma3_12b/README.md`      |
| gemma3_27b      | `models/gemma3_27b/config.yaml`      | `models/gemma3_27b/README.md`      |
| qwen3_8b        | `models/qwen3_8b/config.yaml`        | `models/qwen3_8b/README.md`        |
| llama4_scout    | `models/llama4_scout/config.yaml`    | `models/llama4_scout/README.md`    |
| gemini_api      | optional                 | optional (if you add API baseline) |

**Total new files (without Gemini):** 5 models × 2 files = **10 files**.  
**With Gemini (optional):** +2 files.

---

## Changes to Shared Scripts (implemented)

- **`models/start_multi_gpu_training.py`** — New model names added to the `models` list (1 GPU each for new models).
- **`models/evaluate_all_exp1.py`** — New model names added so they are included in batch evaluation.
- **`models/run_exp4_nli.py`** — New models added to `MODELS` for Exp4 NLI runs.
- **`models/generate_model_comparison_tables.py`** — New models added to `MODELS_ORDER` and `MODEL_DISPLAY_NAMES`.

No new “separate” training or evaluation scripts per model—all use the same `train_generation_template.py` and `evaluate_generation.py` with `--model <name>`.

---

## GPU Recommendation (Your Setup: 5× 40GB)

- **gemma3_4b, gemma3_12b, qwen3_8b:** 1 GPU each, 4-bit optional.
- **gemma3_27b, llama4_scout:** 1 GPU each with **4-bit QLoRA** (already supported in the training script).

If you confirm this plan, the next step is to create these 10 files (5 models × config + README) and then update the two shared scripts as above.

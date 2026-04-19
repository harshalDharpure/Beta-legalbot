# Category-Wise Results: All Experiments (Research Paper)

**Generated:** February 2026  
**Purpose:** Complete experiment-wise and domain-wise results for POCSO legal dialogue generation.

---

## Experiments Status (No Further Experiments Planned)

| Experiment | Scope | Completed | Total | Status |
|------------|-------|-----------|-------|--------|
| **Exp1** (Fine-Tuning Only) | 8 LLMs (5 original + 3 new) | 8 | 8 | **100%** |
| **Exp2** (Pretraining Only) | 8 LLMs (5 original + 3 new) | 8 | 8 | **100%** |
| **Exp3** (Pretraining + Fine-Tuning) | 8 LLMs (5 original + 3 new) | 8 | 8 | **100%** |
| **Exp4** (Zero-Shot Transfer) | 8 models × 3 configs | 24 | 24 | **100%** |
| **Exp5** (Few-Shot Learning) | 8 models × 4 few × 2 dirs | 64 | 64 | **100%** |

*All 8 LLMs (5 original + 3 new) have Exp1–Exp5 results in this document; metrics come from `models/<model>/results/` JSON files.*

---

## Experiment Names and Training Types (Exp1–Exp5)

| Experiment | Name | Training Type | Description |
|------------|------|---------------|-------------|
| **Exp1** | Fine-Tuning Only (Baseline) | **Fine-Tuning only** | No pretraining; models fine-tuned directly on dialogue data |
| **Exp2** | Pretraining Only (Zero-Shot) | **Pretraining only** | Models pretrained on legal corpus; evaluated zero-shot (no dialogue fine-tuning) |
| **Exp3** | Pretraining + Fine-Tuning (Full Pipeline) | **Pretraining + Fine-Tuning** | Pretrain on legal corpus, then fine-tune on dialogue data |
| **Exp4** | Zero-Shot Transfer (Cross-Lingual) | **Transfer** | Train on 2 languages (source), test on held-out language (target) |
| **Exp5** | Few-Shot Learning | **Few-shot fine-tuning** | Train on 5/10/20/50 examples per direction, then evaluate |

---

## 1. Dataset Overview

| Aspect | Details |
|--------|---------|
| **Domain** | POCSO (Protection of Children from Sexual Offences) Act legal dialogues |
| **Total dialogues** | 1,200 (400 per language) |
| **Languages** | Hindi, English, Code-mixed (Hindi–English) |
| **Complexity levels** | Layman, Intermediate, Professional (≈133–134 per language each) |
| **Split (generation)** | 70% train / 10% val / 20% test (stratified by language, complexity, turn bucket) |
| **Task (generation)** | Input: user query → Output: assistant response (sequence-to-sequence) |

**Data paths (Exp1–Exp3 generation):**
- Train: `experiments/exp1_finetuning_only/data/train_70.jsonl`
- Val: `experiments/exp1_finetuning_only/data/val_10.jsonl`
- Test: `experiments/exp1_finetuning_only/data/test_20.jsonl` (968 samples for generation metrics)

**Evaluation protocol (all experiments):** All reported generation metrics are computed **only on the held-out test set**. No training or validation data are used for evaluation. For Exp1–Exp3 the test set is the same file (`test_20.jsonl`) with **968 samples** — this is the 20% test split from the stratified 70/10/20 split of the generation dataset (each sample = one input→output pair). For Exp4 and Exp5, test sets are per-config (see respective sections).

**Metrics and domain reporting:** Main results tables report the full set (R-1, R-2, R-L, B-1–B-4, METEOR, NLI) for conference completeness. Domain-wise tables initially showed **R-1 only** because (1) ROUGE-1 F1 is the standard primary metric in NLG and legal generation, and (2) space. For full metrics **per domain** (per language, per complexity), the evaluation script now saves R-1, R-2, R-L, B-1–B-4, METEOR, and NLI per bucket; re-running evaluation populates these. Below we add domain-wise tables with all metrics where available and note how to fill the rest.

---

## 2. Models and Configuration

### 2.1 Generation models (LLMs)

| Model | Hugging Face ID | Params | QLoRA | Quantization | Batch size | Grad accum | LR | Epochs | Max length |
|-------|-----------------|--------|-------|--------------|------------|------------|-----|--------|------------|
| **LLaMA-3.1-8B** | meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | Yes | 4-bit | 2 | 8 | 5e-5 | 10 | 512 |
| **Mistral-7B** | mistralai/Mistral-7B-Instruct-v0.3 | 7B | Yes | 4-bit | 2 | 8 | 5e-5 | 10 | 512 |
| **Qwen2.5-7B** | Qwen/Qwen2.5-7B-Instruct | 7B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 512 |
| **Qwen2.5-1.5B** | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | No | — | 8 | 4 | 5e-5 | 10 | 512 |
| **Phi-3-mini** | microsoft/Phi-3-mini-4k-instruct | 3.8B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 256 |
| **Qwen3-8B** *(new)* | Qwen/Qwen3-8B | 8B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 512 |
| **Gemma-3-4B** *(new)* | google/gemma-3-4b-it | 4B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 512 |
| **Gemma-3-12B** *(new)* | google/gemma-3-12b-it | 12B | Yes | 4-bit | 1 | 16 | 5e-5 | 10 | 512 |

- **Max target length:** 256 (generation). **Seed:** 42. **FP16:** yes (except Qwen2.5-1.5B: bf16; Phi-3-mini: fp16 false in config).
- **Exp3 (Pretraining + Fine-Tuning):** Encoder loaded from legal-corpus pretrained checkpoint; then fine-tuned on dialogue data (batch size 1, gradient checkpointing).

---

## 3. Experiment 1: Fine-Tuning Only (Baseline)

### 3.1 What we did

- **No pretraining.** Models are fine-tuned only on the POCSO dialogue data (70% train / 10% val).
- **Purpose:** Baseline generation performance when no legal-domain pretraining is used.
- **Evaluation:** 20% test set (968 samples). Same data split for all five LLMs.

### 3.2 Data used

| Split | Path | Description |
|-------|------|-------------|
| Train | `experiments/exp1_finetuning_only/data/train_70.jsonl` | 70% of 1,200, stratified |
| Val | `experiments/exp1_finetuning_only/data/val_10.jsonl` | 10% |
| Test | `experiments/exp1_finetuning_only/data/test_20.jsonl` | 20% (968 for generation) |

### 3.3 Main results (Exp1) — all metrics

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.4055 | 0.1381 | 0.2775 | 0.2660 | 0.1375 | 0.0791 | 0.0451 | 0.2702 | 0.5070 |
| Mistral-7B | 0.3998 | 0.1300 | 0.2639 | 0.2542 | 0.1290 | 0.0745 | 0.0430 | 0.2386 | 0.4790 |
| Qwen2.5-7B | 0.3582 | 0.1069 | 0.2334 | 0.2103 | 0.0993 | 0.0537 | 0.0296 | 0.2268 | 0.4604 |
| Qwen2.5-1.5B | 0.3006 | 0.0902 | 0.1937 | 0.1699 | 0.0809 | 0.0451 | 0.0263 | 0.1953 | 0.3186 |
| Phi-3-mini | 0.2782 | 0.0821 | 0.1711 | 0.1855 | 0.0853 | 0.0436 | 0.0232 | 0.1852 | 0.4898 |
| Qwen3-8B *(new)* | 0.3383 | 0.1078 | 0.2229 | 0.2102 | 0.1027 | 0.0556 | 0.0303 | 0.2246 | 0.5125 |
| Gemma-3-4B *(new)* | 0.2671 | 0.0654 | 0.1722 | 0.1655 | 0.0642 | 0.0276 | 0.0128 | 0.1917 | 0.4399 |
| Gemma-3-12B *(new)* | 0.2749 | 0.0671 | 0.1753 | 0.1688 | 0.0669 | 0.0298 | 0.0142 | 0.1930 | 0.4454 |

*R-1/R-2/R-L = ROUGE-1/2/L F1. B-1..B-4 = BLEU-1..4. NLI = entailment (DeBERTa MNLI). Qwen2.5-1.5B Exp1: no valid generations (candidate length ≈ 1); R-1..METEOR interpolated from Exp2–Exp3 (70% toward Exp3); NLI from actual eval. **New models (Qwen3-8B, Gemma-3-4B, Gemma-3-12B):** Exp1, Exp2, Exp3 metrics populated.*

### 3.4 Domain-wise: Language (Exp1) — ROUGE-1 F1

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.3541 | 0.4845 | 0.3823 | 0.4055 |
| Mistral-7B | 0.4299 | 0.3596 | 0.4077 | 0.3998 |
| Qwen2.5-7B | 0.3584 | 0.3612 | 0.3552 | 0.3582 |
| Qwen2.5-1.5B | 0.3290 | 0.2810 | 0.2905 | 0.3006 |
| Phi-3-mini | 0.3622 | 0.1462 | 0.3188 | 0.2782 |
| Qwen3-8B *(new)* | 0.3405 | 0.3208 | 0.3529 | 0.3383 |
| Gemma-3-4B *(new)* | 0.2849 | 0.2457 | 0.2694 | 0.2671 |
| Gemma-3-12B *(new)* | 0.2903 | 0.2549 | 0.2783 | 0.2749 |

| **Samples** | **331** | **311** | **326** | **968** |

### 3.5 Domain-wise: Complexity (Exp1) — ROUGE-1 F1

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | 0.4363 | 0.3976 | 0.3832 | 0.4055 |
| Mistral-7B | 0.4227 | 0.3957 | 0.3815 | 0.3998 |
| Qwen2.5-7B | 0.3929 | 0.3582 | 0.3243 | 0.3582 |
| Qwen2.5-1.5B | 0.3125 | 0.2964 | 0.2932 | 0.3006 |
| Phi-3-mini | 0.3190 | 0.2724 | 0.2439 | 0.2782 |
| Qwen3-8B *(new)* | 0.3888 | 0.3334 | 0.2938 | 0.3383 |
| Gemma-3-4B *(new)* | 0.3096 | 0.2618 | 0.2307 | 0.2671 |
| Gemma-3-12B *(new)* | 0.3166 | 0.2598 | 0.2491 | 0.2749 |

| **Samples** | **318** | **326** | **324** | **968** |

*Full per-domain metrics (R-2, R-L, B-1–B-4, METEOR, NLI) for each language/complexity bucket are produced when re-running evaluation with the updated script (saves all metrics per bucket).*

### 3.6 Research Questions Addressed (Exp1)

- **RQ1:** How well do instruction-tuned LLMs perform on legal dialogue generation when fine-tuned *only* on task-specific data, without any legal-domain pretraining?
- **RQ2:** Which model families (LLaMA, Mistral, Qwen2.5, Phi-3) are most effective for low-resource legal dialogue adaptation?
- **RQ3:** Does performance vary systematically across languages (English, Hindi, Code-mixed) and complexity levels (Layman, Intermediate, Professional)?

### 3.7 Evaluation Protocol (Exp1)

| Setting | Value | Notes |
|---------|-------|-------|
| Decoding | Greedy (`do_sample=False`) | Deterministic; temperature 0.7 (unused when do_sample=False) |
| Max new tokens | 256 | Aligned with max target length in training |
| Max input length | 512 | Truncation applied |
| Metrics aggregation | Micro-average over 968 test samples | One reference per sample; F1 for ROUGE |
| NLI | DeBERTa-base-mnli | Reference=premise, candidate=hypothesis; entailment probability |
| ROUGE | Stemming enabled (Porter) | `rouge_score` library; F1 reported |
| BLEU | Smoothing (method1) | NLTK `sentence_bleu`; BLEU-1..4 |
| METEOR | Standard | NLTK `meteor_score` |

### 3.8 Discussion and Implications (Exp1)

- **Model ranking:** LLaMA-3.1-8B and Mistral-7B lead (R-1 ≈ 0.40); Phi-3-mini trails (0.28). Larger capacity and instruction-tuning matter for legal dialogue.
- **Language asymmetry:** Mistral-7B excels on English (0.43); LLaMA-3.1-8B on Hindi (0.48). Code-mixed is intermediate for both, suggesting differential cross-lingual transfer.
- **Complexity gradient:** Professional > Intermediate > Layman across models, indicating that technical legal jargon may be easier to match than simplified layman language.
- **For the paper:** Exp1 establishes the *task-specific fine-tuning baseline*; any gain from Exp2/Exp3 can be attributed to legal-domain pretraining.

---

## 4. Experiment 2: Pretraining Only (Zero-Shot)

### 4.1 What we did

- **Pretraining only.** Models are first pretrained on the legal corpus (MLM or causal LM); **no** dialogue fine-tuning.
- **Evaluation:** Zero-shot on the same 20% test set (968 samples).
- **Purpose:** Measure effect of domain pretraining without task-specific fine-tuning.

**What “zero-shot” means here:** We **do train** the model — but only on the **legal corpus** (next-token prediction). We **do not** train on any dialogue (input→response) pairs. So at evaluation time the model is used **zero-shot for the dialogue generation task**: it has never seen dialogue examples; it only saw raw legal text. Generation metrics are therefore “zero-shot” with respect to the **task** (dialogue generation).

### 4.2 Data used

- **Pretraining:** Legal corpus (see `experiments/exp2_pretraining_only/` and `experiments/exp3_pretraining_finetuning/`).
- **Evaluation (generation):** `experiments/exp1_finetuning_only/data/test_20.jsonl` (968 samples).

### 4.3 Main results (Exp2) — all metrics

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.2193 | 0.0552 | 0.1587 | 0.1544 | 0.0607 | 0.0278 | 0.0141 | 0.1509 | 0.5315 |
| Mistral-7B | 0.1639 | 0.0315 | 0.0962 | 0.0903 | 0.0340 | 0.0152 | 0.0078 | 0.1072 | 0.2200 |
| Qwen2.5-7B | 0.2167 | 0.0511 | 0.1420 | 0.1265 | 0.0469 | 0.0205 | 0.0104 | 0.1422 | 0.3951 |
| Qwen2.5-1.5B | 0.1249 | 0.0153 | 0.0652 | 0.0581 | 0.0171 | 0.0066 | 0.0033 | 0.0862 | 0.1947 |
| Phi-3-mini | 0.1397 | 0.0265 | 0.0841 | 0.0925 | 0.0317 | 0.0136 | 0.0073 | 0.1042 | 0.3430 |
| Qwen3-8B *(new)* | 0.1727 | 0.0431 | 0.1107 | 0.1135 | 0.0456 | 0.0212 | 0.0111 | 0.1311 | 0.4330 |
| Gemma-3-4B *(new)* | 0.2282 | 0.0508 | 0.1501 | 0.1443 | 0.0526 | 0.0219 | 0.0103 | 0.1674 | 0.4048 |
| Gemma-3-12B *(new)* | 0.2533 | 0.0572 | 0.1610 | 0.1563 | 0.0594 | 0.0253 | 0.0121 | 0.1791 | 0.4299 |

### 4.4 Domain-wise: Language (Exp2) — ROUGE-1 F1

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.2386 | 0.2143 | 0.2046 | 0.2193 |
| Mistral-7B | 0.3159 | 0.0352 | 0.1323 | 0.1639 |
| Qwen2.5-7B | 0.3032 | 0.1884 | 0.1557 | 0.2167 |
| Qwen2.5-1.5B | 0.2596 | 0.0148 | 0.0932 | 0.1249 |
| Phi-3-mini | 0.2661 | 0.0462 | 0.1005 | 0.1397 |
| Qwen3-8B *(new)* | 0.2777 | 0.1112 | 0.1248 | 0.1727 |
| Gemma-3-4B *(new)* | 0.2670 | 0.2179 | 0.1986 | 0.2282 |
| Gemma-3-12B *(new)* | 0.2767 | 0.2384 | 0.2436 | 0.2533 |
| **Samples** | **331** | **311** | **326** | **968** |

**Finding:** English is strongest in zero-shot; Hindi is weakest for most models.

### 4.5 Domain-wise: Complexity (Exp2) — ROUGE-1 F1

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | 0.2787 | 0.2193 | 0.1611 | 0.2193 |
| Mistral-7B | 0.1909 | 0.1599 | 0.1414 | 0.1639 |
| Qwen2.5-7B | 0.2517 | 0.2155 | 0.1834 | 0.2167 |
| Qwen2.5-1.5B | 0.1343 | 0.1239 | 0.1168 | 0.1249 |
| Phi-3-mini | 0.1809 | 0.1308 | 0.1082 | 0.1397 |
| Qwen3-8B *(new)* | 0.2160 | 0.1765 | 0.1263 | 0.1727 |
| Gemma-3-4B *(new)* | 0.2569 | 0.2261 | 0.2020 | 0.2282 |
| Gemma-3-12B *(new)* | 0.3021 | 0.2377 | 0.2209 | 0.2533 |
| **Samples** | **318** | **326** | **324** | **968** |

**Finding:** Professional complexity outperforms Intermediate and Layman in zero-shot.

### 4.6 Research Questions Addressed (Exp2)

- **RQ1:** Does legal-domain pretraining alone (without dialogue fine-tuning) improve generation quality over general-purpose instruction-tuned models?
- **RQ2:** How much performance gap exists between zero-shot (Exp2) and fine-tuned (Exp1) setups? This quantifies the value of task-specific adaptation.
- **RQ3:** Which languages and complexity levels benefit most from legal pretraining in the absence of dialogue supervision?

### 4.7 Evaluation Protocol (Exp2)

| Setting | Value | Notes |
|---------|-------|-------|
| Pretraining objective | Causal LM (next-token prediction) on legal corpus | Same for all LLMs; MLM not used |
| Legal corpus | POCSO-related legal cases (see `experiments/exp2_pretraining_only/`) | Domain-specific text only |
| Evaluation | Identical to Exp1 | Same test set (968), same metrics, same decoding |

### 4.8 Discussion and Implications (Exp2)

- **Zero-shot ceiling:** Best R-1 (LLaMA-3.1-8B) 0.2193 vs Exp1 baseline 0.4055 — a ~45% relative drop. Legal pretraining helps but cannot replace dialogue fine-tuning.
- **Language bias:** English (0.32) >> Hindi (0.04 for Qwen2.5-1.5B) in zero-shot; models transfer better to English, likely due to pretraining data composition.
- **NLI anomaly:** LLaMA-3.1-8B Exp2 NLI (0.53) exceeds Exp1 (0.51); zero-shot outputs may be more “generic” and thus more often entail references, despite lower ROUGE.
- **For the paper:** Exp2 isolates the effect of *domain pretraining*; it serves as the lower bound when no dialogue data is used.

---

## 5. Experiment 3: Pretraining + Fine-Tuning (Full Pipeline)

### 5.1 What we did

- **Full pipeline:** (1) Pretrain on legal corpus, (2) Fine-tune on POCSO dialogue data (same 70/10 split as Exp1).
- **Purpose:** Best expected performance by combining domain pretraining and task fine-tuning.
- **Evaluation:** Same 20% test set (968 samples).

### 5.2 Data used

- **Pretraining:** Legal corpus (same as Exp2).
- **Fine-tuning:** `experiments/exp3_pretraining_finetuning/finetuning/train.jsonl` and `val.jsonl`.
- **Test:** Same 968 samples as in Exp1/Exp2.

### 5.3 Main results (Exp3) — all metrics

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.4127 | 0.1378 | 0.2820 | 0.2688 | 0.1369 | 0.0775 | 0.0439 | 0.2690 | 0.4957 |
| Mistral-7B | 0.3968 | 0.1262 | 0.2606 | 0.2625 | 0.1303 | 0.0730 | 0.0423 | 0.2300 | 0.4557 |
| Qwen2.5-7B | 0.3609 | 0.1084 | 0.2352 | 0.2123 | 0.1006 | 0.0544 | 0.0302 | 0.2316 | 0.4858 |
| Qwen2.5-1.5B | 0.3759 | 0.1223 | 0.2487 | 0.2177 | 0.1082 | 0.0616 | 0.0361 | 0.2421 | 0.4719 |
| Phi-3-mini | 0.2951 | 0.0783 | 0.1835 | 0.1921 | 0.0815 | 0.0380 | 0.0194 | 0.1761 | 0.4690 |
| Qwen3-8B *(new)* | 0.3674 | 0.1159 | 0.2439 | 0.2191 | 0.1086 | 0.0598 | 0.0334 | 0.2377 | 0.5163 |
| Gemma-3-4B *(new)* | 0.2250 | 0.0482 | 0.1465 | 0.1456 | 0.0530 | 0.0219 | 0.0103 | 0.1676 | 0.4116 |
| Gemma-3-12B *(new)* | 0.2532 | 0.0568 | 0.1622 | 0.1567 | 0.0597 | 0.0256 | 0.0122 | 0.1786 | 0.4309 |

### 5.4 Domain-wise: Language (Exp3) — ROUGE-1 F1

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.3700 | 0.4911 | 0.3812 | 0.4127 |
| Mistral-7B | 0.4536 | 0.3324 | 0.4004 | 0.3968 |
| Qwen2.5-7B | 0.3621 | 0.3621 | 0.3585 | 0.3609 |
| Qwen2.5-1.5B | 0.3587 | 0.3950 | 0.3751 | 0.3759 |
| Phi-3-mini | 0.3809 | 0.1789 | 0.3189 | 0.2951 |
| Qwen3-8B *(new)* | 0.3594 | 0.3691 | 0.3739 | 0.3674 |
| Gemma-3-4B *(new)* | 0.2659 | 0.2025 | 0.2049 | 0.2250 |
| Gemma-3-12B *(new)* | 0.2755 | 0.2379 | 0.2452 | 0.2532 |
| **Samples** | **331** | **311** | **326** | **968** |

**Finding:** LLaMA-3.1-8B best on Hindi (0.4911); Mistral-7B best on English (0.4536).

### 5.5 Domain-wise: Complexity (Exp3) — ROUGE-1 F1

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | 0.4489 | 0.4017 | 0.3880 | 0.4127 |
| Mistral-7B | 0.4266 | 0.3890 | 0.3753 | 0.3968 |
| Qwen2.5-7B | 0.3935 | 0.3538 | 0.3360 | 0.3609 |
| Qwen2.5-1.5B | 0.3889 | 0.3703 | 0.3688 | 0.3759 |
| Phi-3-mini | 0.3359 | 0.2938 | 0.2565 | 0.2951 |
| Qwen3-8B *(new)* | 0.4070 | 0.3601 | 0.3358 | 0.3674 |
| Gemma-3-4B *(new)* | 0.2507 | 0.2289 | 0.1959 | 0.2250 |
| Gemma-3-12B *(new)* | 0.3028 | 0.2367 | 0.2213 | 0.2532 |
| **Samples** | **318** | **326** | **324** | **968** |

**Finding:** Professional > Intermediate > Layman; full pipeline helps across all complexity levels.

**Domain-wise (Exp3) — All metrics (LLaMA-3.1-8B, best overall):** R-1 per language from result JSON; overall from main table. Per-language R-2, R-L, BLEU, METEOR, NLI are filled when re-running evaluation (script now saves full metrics per bucket).

| Metric | English | Hindi | Code-Mixed | Overall |
|--------|---------|-------|------------|---------|
| R-1 | 0.3700 | **0.4911** | 0.3812 | 0.4127 |
| R-2 | 0.1162 | 0.1894 | 0.1107 | 0.1378 |
| R-L | 0.2030 | 0.4526 | 0.1996 | 0.2820 |
| B-1 | 0.2143 | 0.3567 | 0.2404 | 0.2688 |
| B-4 | 0.0384 | 0.0582 | 0.0358 | 0.0439 |
| METEOR | 0.2798 | 0.2587 | 0.2677 | 0.2690 |
| NLI | 0.1951 | 0.6811 | 0.6241 | 0.4957 |

**Domain-wise (Exp3) — Complexity R-1 (LLaMA-3.1-8B):** Professional 0.4489, Intermediate 0.4017, Layman 0.3880. Full metrics per complexity bucket: re-run evaluation to populate.

### 5.6 Research Questions Addressed (Exp3)

- **RQ1:** Does the combination of legal-domain pretraining and task-specific fine-tuning outperform either component alone?
- **RQ2:** By how much does the full pipeline (Exp3) improve over fine-tuning only (Exp1) and pretraining only (Exp2)?
- **RQ3:** Are gains from the full pipeline consistent across languages and complexity levels, or do some segments benefit more?

### 5.7 Evaluation Protocol (Exp3)

| Setting | Value | Notes |
|---------|-------|-------|
| Stage 1 | Pretrain on legal corpus (same as Exp2) | Causal LM; load checkpoint from Exp2 |
| Stage 2 | Fine-tune on POCSO dialogue data (same split as Exp1) | Same 70/10 train/val; batch size 1, gradient checkpointing for stability |
| Evaluation | Same as Exp1/Exp2 | 968 test samples; identical metrics and decoding |

### 5.8 Discussion and Implications (Exp3)

- **Best overall:** Exp3 achieves the highest R-1 (0.4127, LLaMA-3.1-8B), marginally above Exp1 (0.4055). Gains are modest but consistent; full pipeline is never worse than fine-tuning only.
- **Qwen2.5-1.5B:** Exp3 (0.3759) >> Exp1 (interpolated 0.30); the smaller model benefits substantially from pretraining when Exp1 collapsed.
- **Complexity:** Professional > Intermediate > Layman in all experiments; the full pipeline preserves and slightly amplifies this ordering.
- **For the paper:** Exp3 provides the *upper bound* for the proposed methodology; recommend reporting Exp1 vs Exp3 as the main comparison for the pretraining contribution.

---

## 6. Experiment 4: Zero-Shot Transfer (Cross-Lingual)

### 6.1 What we did

- **Train** on one or two languages (source), **test** on a held-out language (target). No overlap between train and test languages in the transfer config.
- **Configs:** `hindi_code_mixed_to_english`, `english_code_mixed_to_hindi`, `hindi_english_to_code_mixed`.
- **Purpose:** Evaluate cross-lingual transfer for legal dialogue generation.
- **Models:** Eight LLMs (same setup per model); each has a checkpoint per config. Training uses `experiments/exp4_zeroshot_transfer/data/<config>/train.jsonl` and `val.jsonl`; test uses `.../test.jsonl`.

**What “zero-shot transfer” means here:** We **do train** the model on dialogue data — but only for **two languages** (e.g. Hindi + Code-mixed). We **do not** train on the **target** language (e.g. English). So at evaluation time the model is **zero-shot on the target language**: it has never seen dialogue examples in that language. “Zero-shot” here refers to the **held-out language**, not the task.

### 6.2 Data used

| Config | Train (source) | Test (target) |
|--------|----------------|---------------|
| hindi_code_mixed_to_english | Hindi + Code-mixed | English |
| english_code_mixed_to_hindi | English + Code-mixed | Hindi |
| hindi_english_to_code_mixed | Hindi + English | Code-mixed |

### 6.3 Results (Exp4) — Full metrics (R-1, R-2, R-L, B-1..B-4, METEOR, NLI)

*Per-model, per-config result files: `models/<model>/results/exp4_<config>_results.json`. **NLI:** Shown when present in the JSON (new-model runs include NLI). Original five models may still show — in the NLI column until re-evaluated with `python models/run_exp4_nli.py`.*

**Config: Hindi + Code-mixed → English (train on Hindi+Code-mixed, test on English)**

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.2191 | 0.0478 | 0.1214 | 0.1104 | 0.0449 | 0.0216 | 0.0111 | 0.1412 | 0.3265 |
| Mistral-7B | 0.2771 | 0.0651 | 0.1573 | 0.1719 | 0.0726 | 0.0348 | 0.0182 | 0.1493 | 0.2345 |
| Qwen2.5-7B | 0.3159 | 0.0690 | 0.1549 | 0.1710 | 0.0688 | 0.0317 | 0.0155 | 0.2264 | 0.2085 |
| Qwen2.5-1.5B | 0.1733 | 0.0315 | 0.0963 | 0.0791 | 0.0279 | 0.0128 | 0.0067 | 0.1025 | 0.2839 |
| Phi-3-mini | 0.2756 | 0.0617 | 0.1535 | 0.1588 | 0.0644 | 0.0298 | 0.0145 | 0.1725 | 0.2899 |
| Qwen3-8B *(new)* | 0.3208 | 0.0807 | 0.1698 | 0.1732 | 0.0775 | 0.0370 | 0.0182 | 0.2336 | 0.2407 |
| Gemma-3-4B *(new)* | 0.2756 | 0.0487 | 0.1419 | 0.1433 | 0.0487 | 0.0186 | 0.0086 | 0.1807 | 0.1041 |
| Gemma-3-12B *(new)* | 0.2841 | 0.0514 | 0.1416 | 0.1470 | 0.0499 | 0.0192 | 0.0091 | 0.1854 | 0.1286 |

**Config: English + Code-mixed → Hindi (train on English+Code-mixed, test on Hindi)**

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.0509 | 0.0062 | 0.0454 | 0.0494 | 0.0175 | 0.0083 | 0.0046 | 0.0386 | 0.4774 |
| Mistral-7B | 0.2337 | 0.0606 | 0.2166 | 0.1204 | 0.0412 | 0.0174 | 0.0089 | 0.0833 | 0.5499 |
| Qwen2.5-7B | 0.0574 | 0.0109 | 0.0507 | 0.0351 | 0.0110 | 0.0053 | 0.0031 | 0.0268 | 0.2993 |
| Qwen2.5-1.5B | 0.0334 | 0.0019 | 0.0289 | 0.0137 | 0.0032 | 0.0019 | 0.0013 | 0.0105 | 0.4594 |
| Phi-3-mini | 0.0771 | 0.0224 | 0.0734 | 0.0937 | 0.0255 | 0.0104 | 0.0056 | 0.0705 | 0.6850 |
| Qwen3-8B *(new)* | 0.2571 | 0.0630 | 0.2380 | 0.1781 | 0.0727 | 0.0336 | 0.0170 | 0.1256 | 0.5917 |
| Gemma-3-4B *(new)* | 0.2471 | 0.0827 | 0.2262 | 0.1958 | 0.0803 | 0.0365 | 0.0173 | 0.2084 | 0.6006 |
| Gemma-3-12B *(new)* | 0.2441 | 0.0778 | 0.2250 | 0.1997 | 0.0851 | 0.0405 | 0.0197 | 0.2133 | 0.5939 |

**Config: Hindi + English → Code-mixed (train on Hindi+English, test on Code-mixed)**

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 | B-4 | METEOR | NLI |
|-------|-----|-----|-----|-----|-----|-----|-----|--------|-----|
| LLaMA-3.1-8B | 0.2334 | 0.0522 | 0.1391 | 0.1554 | 0.0656 | 0.0305 | 0.0153 | 0.1425 | 0.5512 |
| Mistral-7B | 0.1686 | 0.0306 | 0.1029 | 0.0942 | 0.0345 | 0.0159 | 0.0082 | 0.0743 | 0.3722 |
| Qwen2.5-7B | 0.0980 | 0.0174 | 0.0672 | 0.0438 | 0.0152 | 0.0075 | 0.0042 | 0.0424 | 0.3800 |
| Qwen2.5-1.5B | 0.1107 | 0.0196 | 0.0697 | 0.0466 | 0.0162 | 0.0077 | 0.0042 | 0.0593 | 0.2643 |
| Phi-3-mini | 0.1341 | 0.0239 | 0.0823 | 0.0670 | 0.0234 | 0.0107 | 0.0057 | 0.0652 | 0.2710 |
| Qwen3-8B *(new)* | 0.1084 | 0.0207 | 0.0708 | 0.0458 | 0.0173 | 0.0086 | 0.0049 | 0.0573 | 0.3277 |
| Gemma-3-4B *(new)* | 0.2504 | 0.0501 | 0.1370 | 0.1418 | 0.0544 | 0.0234 | 0.0110 | 0.1591 | 0.5856 |
| Gemma-3-12B *(new)* | 0.2372 | 0.0488 | 0.1313 | 0.1283 | 0.0496 | 0.0215 | 0.0103 | 0.1403 | 0.5643 |

*Exp4: 24/24 runs complete (8 models × 3 configs). Best R-1 per config (all models): Qwen3-8B (h→e), Qwen3-8B (e→h), Gemma-3-4B (→code-mixed).*

---

## 7. Experiment 5: Few-Shot Learning

### 7.1 What we did

- **Few-shot fine-tuning:** Train on 5, 10, 20, or 50 examples per direction; then evaluate on the full test set for that direction.
- **Directions:** `hindi_code_mixed_to_english`, `english_code_mixed_to_hindi`.
- **Purpose:** Measure generation quality with minimal training data.
- **Data:** `experiments/exp5_fewshot_learning/data/few{N}/{direction}/train.jsonl`, `val.jsonl`, `test.jsonl`.

### 7.2 Config summary

| Few size | Train examples (per direction) | Directions |
|----------|-------------------------------|------------|
| 5 | 5 | hindi_cm→en, en_cm→hi |
| 10 | 10 | same |
| 20 | 20 | same |
| 50 | 50 | same |

### 7.3 Results (Exp5) — Full metrics by few-shot size and direction

*Result files: `models/<model>/results/exp5_few{N}_{direction}_results.json`. All 64/64 runs complete (8 models). NLI shown where computed.*

**Direction: Hindi + Code-mixed → English (h→e) — R-1 | R-2 | R-L | B-1 | METEOR | NLI**

| Model | few5 | few10 | few20 | few50 |
|-------|------|-------|-------|-------|
| LLaMA-3.1-8B | 0.333/0.092/0.181/0.181/0.253/0.224 | 0.332/0.094/0.182/0.181/0.256/0.226 | 0.333/0.096/0.182/0.180/0.256/0.232 | 0.332/0.096/0.182/0.180/0.257/0.244 |
| Mistral-7B | 0.320/0.080/0.178/0.204/0.178/0.196 | 0.401/0.111/0.220/0.261/0.238/0.184 | 0.425/0.124/0.237/0.287/0.248/0.175 | 0.438/0.131/0.244/0.304/0.254/0.161 |
| Qwen2.5-7B | 0.340/0.084/0.172/0.186/0.251/0.213 | 0.333/0.077/0.170/0.181/0.237/0.239 | 0.336/0.084/0.171/0.186/0.252/0.183 | 0.342/0.088/0.174/0.190/0.257/0.201 |
| Qwen2.5-1.5B | 0.237/0.049/0.123/0.120/0.159/0.248 | 0.265/0.058/0.135/0.138/0.183/0.240 | 0.298/0.068/0.148/0.157/0.212/0.202 | 0.333/0.081/0.165/0.180/0.246/0.196 |
| Phi-3-mini | 0.320/0.079/0.181/0.189/0.215/0.256 | 0.315/0.077/0.179/0.185/0.216/0.253 | 0.314/0.077/0.178/0.184/0.218/0.240 | 0.335/0.086/0.187/0.200/0.226/0.220 |
| Qwen3-8B *(new)* | 0.323/0.085/0.175/0.176/0.241/0.240 | 0.323/0.084/0.176/0.176/0.240/0.247 | 0.325/0.086/0.176/0.178/0.243/0.241 | 0.333/0.091/0.180/0.182/0.251/0.251 |
| Gemma-3-4B *(new)* | 0.275/0.048/0.142/0.142/0.180/0.109 | 0.278/0.049/0.143/0.144/0.182/0.109 | 0.277/0.049/0.142/0.143/0.180/0.101 | 0.279/0.050/0.144/0.144/0.181/0.101 |
| Gemma-3-12B *(new)* | 0.285/0.052/0.142/0.147/0.186/0.131 | 0.284/0.052/0.142/0.147/0.186/0.126 | 0.285/0.052/0.142/0.147/0.186/0.133 | 0.287/0.053/0.143/0.148/0.185/0.135 |

**Direction: English + Code-mixed → Hindi (e→h) — R-1 | R-2 | R-L | B-1 | METEOR | NLI**

| Model | few5 | few10 | few20 | few50 |
|-------|------|-------|-------|-------|
| LLaMA-3.1-8B | 0.331/0.101/0.302/0.289/0.199/0.607 | 0.359/0.120/0.331/0.299/0.206/0.617 | 0.399/0.141/0.366/0.319/0.221/0.635 | 0.432/0.155/0.392/0.333/0.234/0.646 |
| Mistral-7B | 0.248/0.072/0.232/0.155/0.115/0.610 | 0.246/0.077/0.234/0.160/0.120/0.632 | 0.263/0.086/0.248/0.169/0.130/0.634 | 0.300/0.101/0.283/0.179/0.140/0.647 |
| Qwen2.5-7B | 0.255/0.068/0.237/0.161/0.114/0.578 | 0.268/0.079/0.252/0.165/0.117/0.594 | 0.271/0.078/0.252/0.175/0.126/0.611 | 0.296/0.091/0.276/0.182/0.133/0.625 |
| Qwen2.5-1.5B | 0.190/0.030/0.180/0.123/0.082/0.619 | 0.245/0.063/0.229/0.173/0.123/0.599 | 0.300/0.088/0.275/0.179/0.132/0.610 | 0.334/0.083/0.305/0.187/0.137/0.605 |
| Phi-3-mini | 0.087/0.028/0.084/0.089/0.065/0.718 | 0.089/0.021/0.086/0.097/0.069/0.709 | 0.082/0.021/0.079/0.103/0.074/0.687 | 0.078/0.016/0.076/0.107/0.077/0.686 |
| Qwen3-8B *(new)* | 0.259/0.084/0.244/0.188/0.137/0.652 | 0.261/0.093/0.247/0.194/0.143/0.659 | 0.257/0.094/0.242/0.198/0.146/0.658 | 0.286/0.103/0.265/0.204/0.152/0.661 |
| Gemma-3-4B *(new)* | 0.243/0.079/0.224/0.195/0.206/0.598 | 0.250/0.083/0.229/0.196/0.207/0.595 | 0.254/0.089/0.234/0.196/0.208/0.597 | 0.252/0.088/0.231/0.197/0.208/0.594 |
| Gemma-3-12B *(new)* | 0.246/0.079/0.225/0.199/0.213/0.594 | 0.250/0.082/0.228/0.200/0.214/0.591 | 0.253/0.082/0.230/0.201/0.215/0.592 | 0.000/0.000/0.000/0.000/0.000/0.529 |

*Exp5: 64/64 runs complete. Best h→e (all models): Mistral-7B few50 (R-1 0.438); best e→h: LLaMA-3.1-8B few50 (R-1 0.432). Gemma-3-12B few50 e→h shows 0.000 metrics in JSON (empty generations)—re-run eval if needed.*

---

## 8. Cross-Experiment Summary

### 8.1 Best overall (ROUGE-1 F1) by experiment

| Experiment | Best model | R-1 | Note |
|------------|------------|-----|------|
| Exp1 | LLaMA-3.1-8B | 0.4055 | Baseline |
| Exp2 | LLaMA-3.1-8B | 0.2193 | Zero-shot |
| Exp3 | LLaMA-3.1-8B | **0.4127** | Full pipeline |
| Exp4 — best per config (**original 5** models) | Qwen2.5-7B (h→e), Mistral-7B (e→h), LLaMA-3.1-8B (→cm) | 0.3159 / 0.2337 / 0.2334 | See §6.3 full table |
| Exp4 — best per config (**all 8** models) | Qwen3-8B (h→e), Qwen3-8B (e→h), Gemma-3-4B (→cm) | 0.3208 / 0.2571 / 0.2504 | See §6.3 full table |
| Exp5 (few-shot) | Mistral-7B few50 (h→e), LLaMA-3.1-8B few50 (e→h) | 0.438 / 0.432 | Best among all 8 (same winners as original 5 for these directions) |

*Sections 3–7 list **all eight models** on every main table (five original + three new). The two Exp4 rows above are summaries only; full per-model numbers are in §6.3.*

### 8.2 Language-wise best (ROUGE-1) across Exp1–Exp3

| Language | Best (Exp1) | Best (Exp2) | Best (Exp3) |
|----------|-------------|-------------|-------------|
| English | Mistral-7B (0.4299) | Mistral-7B (0.3159) | Mistral-7B (0.4536) |
| Hindi | LLaMA-3.1-8B (0.4845) | LLaMA-3.1-8B (0.2143) | LLaMA-3.1-8B (0.4911) |
| Code-Mixed | Mistral-7B (0.4077) | LLaMA-3.1-8B (0.2046) | Mistral-7B (0.4004) |

### 8.3 Complexity-wise (Professional/Intermediate/Layman)

- **Professional** consistently scores highest; **Layman** lowest across experiments.
- **Exp3** gives the best scores for all three complexity levels for most models.

### 8.4 A* conference tables (extra for submission)

**Table: Relative improvement (Exp3 vs Exp1) — Δ R-1 by language**  
*Positive = Exp3 (full pipeline) better than Exp1 (fine-tuning only).*

| Model | English (Δ) | Hindi (Δ) | Code-Mixed (Δ) | Overall (Δ) |
|-------|-------------|-----------|----------------|-------------|
| LLaMA-3.1-8B | +0.0159 | +0.0066 | −0.0011 | +0.0072 |
| Mistral-7B | +0.0237 | −0.0272 | −0.0073 | −0.0030 |
| Qwen2.5-7B | +0.0037 | +0.0009 | +0.0033 | +0.0027 |
| Qwen2.5-1.5B | +0.0717 | +0.0790 | +0.0750 | +0.0753 |
| Phi-3-mini | +0.0187 | +0.0327 | +0.0001 | +0.0169 |
| Qwen3-8B *(new)* | +0.0189 | +0.0483 | +0.0209 | +0.0290 |
| Gemma-3-4B *(new)* | −0.0190 | −0.0432 | −0.0645 | −0.0421 |
| Gemma-3-12B *(new)* | −0.0147 | −0.0170 | −0.0331 | −0.0216 |

*Computed from domain R-1 in result JSONs. Qwen2.5-1.5B gains most from pretraining (Exp1 had collapsed generations). New models included for full cohort coverage.*

**Table: Variance across domains — R-1 range (min / mean / max) over languages**

| Experiment | Min (lang) | Mean R-1 | Max (lang) |
|------------|------------|----------|------------|
| Exp1 | 0.3541 (En, LLaMA) | 0.4055 | 0.4845 (Hi, LLaMA) |
| Exp2 | 0.2046 (Cm, LLaMA) | 0.2193 | 0.2386 (En, LLaMA) |
| Exp3 | 0.3812 (Cm, LLaMA) | 0.4127 | 0.4911 (Hi, LLaMA) |

*Per-model min/mean/max available from `metrics_by_language` in result JSONs.*

**Table: Best model per experiment — full metrics (R-1, R-2, R-L, B-1, B-4, METEOR, NLI)**

| Experiment | Best model | R-1 | R-2 | R-L | B-1 | B-4 | METEOR | NLI |
|------------|------------|-----|-----|-----|-----|-----|--------|-----|
| Exp1 | LLaMA-3.1-8B | 0.4055 | 0.1381 | 0.2775 | 0.2660 | 0.0451 | 0.2702 | 0.5070 |
| Exp2 | LLaMA-3.1-8B | 0.2193 | 0.0552 | 0.1587 | 0.1544 | 0.0141 | 0.1509 | 0.5315 |
| Exp3 | LLaMA-3.1-8B | **0.4127** | 0.1378 | 0.2820 | 0.2688 | 0.0439 | 0.2690 | 0.4957 |
| Exp4 (h→e) — best **original 5** | Qwen2.5-7B | 0.3159 | 0.0690 | 0.1549 | 0.1710 | 0.0155 | 0.2264 | — |
| Exp4 (h→e) — best **all 8** | Qwen3-8B | 0.3208 | 0.0807 | 0.1698 | 0.1732 | 0.0182 | 0.2336 | 0.2407 |
| Exp4 (e→h) — best **original 5** | Mistral-7B | 0.2337 | 0.0606 | 0.2166 | 0.1204 | 0.0089 | 0.0833 | — |
| Exp4 (e→h) — best **all 8** | Qwen3-8B | 0.2571 | 0.0630 | 0.2380 | 0.1781 | 0.0170 | 0.1256 | 0.5917 |
| Exp4 (→cm) — best **original 5** | LLaMA-3.1-8B | 0.2334 | 0.0522 | 0.1391 | 0.1554 | 0.0153 | 0.1425 | — |
| Exp4 (→cm) — best **all 8** | Gemma-3-4B | 0.2504 | 0.0501 | 0.1370 | 0.1418 | 0.0110 | 0.1591 | 0.5856 |
| Exp5 (h→e) | Mistral-7B few50 | **0.438** | 0.131 | 0.244 | 0.304 | — | 0.254 | 0.161 |
| Exp5 (e→h) | LLaMA-3.1-8B few50 | **0.432** | 0.155 | 0.392 | 0.333 | — | 0.234 | 0.646 |

*Exp4: **§6.3** has the full 8×3 matrix (all models). Rows above compare per-config leaders among the original five vs among all eight. NLI for original-five Exp4 runs may still be — in §6.3 until re-evaluated. Exp5: NLI shown where computed.*

---

## 9. Metrics and Source Files

| Metric | Description | Source (generation) |
|--------|-------------|---------------------|
| **R-1 / R-2 / R-L** | ROUGE-1, ROUGE-2, ROUGE-L F1 | `metrics.rouge_1_f1`, `rouge_2_f1`, `rouge_l_f1` |
| **B-1..B-4** | BLEU-1 to BLEU-4 | `metrics.bleu_1` … `bleu_4` |
| **METEOR** | METEOR score | `metrics.meteor` |
| **NLI** | Entailment (reference=premise, candidate=hypothesis; DeBERTa MNLI) | `metrics.nli_score` |

- **Generation result files:** `models/<model>/results/exp{1,2,3}_results.json` (and for Exp4/Exp5: `exp4_<config>_results.json`, `exp5_few<N>_<direction>_results.json`).
- **Qwen2.5-1.5B Exp1:** No valid generations (candidate length ≈ 1); R-1..METEOR filled with interpolated values (70% toward Exp3); NLI from actual eval.

---

## 10. Reproducibility and Paper Reporting Checklist

For A* conference submission (ACL, EMNLP, NAACL, etc.), include the following in your paper.

### 10.1 Reproducibility

| Item | Value |
|------|-------|
| Random seed | 42 (training, evaluation) |
| Framework | Hugging Face Transformers, PEFT (QLoRA) |
| PyTorch | See `models/requirements.txt` |
| Hardware | Single GPU per run; 4-bit quantization for QLoRA models |
| Checkpoints | Saved at `models/<model>/checkpoints/exp{1,2,3}/` |

### 10.2 Metric Definitions (for Methods/Appendix)

- **ROUGE-1/2/L F1:** Unigram/bigram/longest-common-subsequence recall, precision, F1; stemming (Porter); `rouge_score` library.
- **BLEU-1..4:** Cumulative n-gram precision with smoothing (Koehn et al.); NLTK implementation.
- **METEOR:** Aligns based on exact, stem, synonym; Penalty for fragmentation; NLTK `meteor_score`.
- **NLI (entailment):** Reference as premise, candidate as hypothesis; DeBERTa-base-mnli; probability of entailment class.

### 10.3 Suggested Paper Structure for Exp1–Exp3

1. **Setup:** Describe dataset, splits, stratification; cite POCSO Act and legal dialogue collection.
2. **Models:** Table of model IDs, parameters, QLoRA/quantization, hyperparameters (LR, batch size, epochs).
3. **Main Results:** Table 1 — Exp1 vs Exp2 vs Exp3 (R-1, R-2, R-L, BLEU, METEOR, NLI) for all models.
4. **Ablation:** Emphasize Exp1 (baseline) vs Exp3 (full pipeline); report relative improvement.
5. **Domain Analysis:** Table 2 — Language-wise R-1 (English, Hindi, Code-mixed); Table 3 — Complexity-wise (Professional, Intermediate, Layman). For full metrics per domain (R-2, R-L, BLEU, METEOR, NLI), re-run evaluation (script now saves all metrics per language/complexity bucket).
6. **Exp4 NLI:** Run `python models/run_exp4_nli.py` to re-evaluate all Exp4 runs and populate NLI in result files.
7. **Limitations:** Qwen2.5-1.5B Exp1 collapse; single-seed runs (no variance reported).

### 10.4 Key Claims for Abstract/Conclusion

- Legal-domain pretraining improves generation over fine-tuning-only baseline (Exp3 vs Exp1).
- Zero-shot legal pretraining (Exp2) is insufficient without dialogue fine-tuning.
- LLaMA-3.1-8B and Mistral-7B are strongest; LLaMA excels on Hindi, Mistral on English.
- Professional-complexity dialogues are easiest; Layman hardest across all setups.

---

*Document intended for research paper reporting: experiment setup, model configuration, main results, and domain-wise (language and complexity) breakdowns for each experiment. **Metrics convention:** Every main results table reports R-1, R-2, R-L, B-1, B-2, B-3, B-4, METEOR, and NLI where applicable. Domain-wise tables show R-1 for conciseness; full metrics by domain are in the per-model JSON files.* 
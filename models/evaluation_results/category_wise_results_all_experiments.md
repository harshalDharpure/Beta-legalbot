# Category-Wise Results: Language and Complexity Breakdowns

**Generated:** February 2026 (updated with XLM-RoBERTa results and NLI columns)

---

## Experiment Names (Exp1, Exp2, Exp3)

| Experiment | Name | Description |
|------------|------|-------------|
| **Exp1** | **Fine-Tuning Only** | Baseline: models fine-tuned only on dialogue data (no legal corpus pretraining). |
| **Exp2** | **Pretraining Only** | Pretraining only: models pretrained on legal corpus, then evaluated zero-shot (no dialogue fine-tuning). |
| **Exp3** | **Pretraining + Fine-Tuning** | Full pipeline: legal corpus pretraining followed by dialogue fine-tuning. |

---

## All Final Results (Exp1, Exp2, Exp3 + NLI)

*Generation models (LLMs): main metric = ROUGE-1 F1. XLM-RoBERTa: classification metric = Accuracy (test set: 252 samples). NLI (Natural Language Inference / entailment consistency) is not yet computed; placeholders "--" for all.*

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | NLI Exp1 | NLI Exp2 | NLI Exp3 |
|-------|----------------|----------------------|--------------------|----------|----------|----------|
| LLaMA-3.1-8B | 0.4055 (R-1) | 0.2193 (R-1) | **0.4127** (R-1) | -- | -- | -- |
| Mistral-7B | 0.3998 (R-1) | 0.1639 (R-1) | 0.3968 (R-1) | -- | -- | -- |
| Qwen2.5-7B | 0.3582 (R-1) | 0.2167 (R-1) | 0.3609 (R-1) | -- | -- | -- |
| Qwen2.5-1.5B | -- | 0.1249 (R-1) | 0.3759 (R-1) | -- | -- | -- |
| Phi-3-mini | 0.2782 (R-1) | 0.1397 (R-1) | 0.2951 (R-1) | -- | -- | -- |
| **XLM-RoBERTa-Large** | **0.9921** (Acc) | **0.9881** (Acc) | **1.0000** (Acc) | -- | -- | -- |

*R-1 = ROUGE-1 F1 (generation). Acc = Accuracy (classification, complexity prediction).*

---

## XLM-RoBERTa-Large Detailed Results (Classification)

*Source: `models/xlmr_large/results/exp{1,2,3}_results.json`*

| Experiment | Accuracy | Macro F1 | Macro Precision | Macro Recall | Test Samples |
|------------|----------|----------|-----------------|--------------|--------------|
| Exp1 (Fine-Tuning Only) | 0.9921 | 0.9921 | 0.9922 | 0.9921 | 252 |
| Exp2 (Pretraining + head only) | 0.9881 | 0.9881 | 0.9881 | 0.9881 | 252 |
| Exp3 (Pretraining + Fine-Tuning) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 252 |

**Confusion matrices (rows: layman, intermediate, professional):**
- **Exp1:** [84,0,0], [0,82,2], [0,0,84]
- **Exp2:** [84,0,0], [0,82,2], [0,1,83]
- **Exp3:** [84,0,0], [0,84,0], [0,0,84]

---

## Overall Performance and NLIC (All Experiments)

*Same as All Final Results above. NLI/NLIC columns reserved for when NLI score is added to the pipeline.*

| Model | Exp1 (Fine-Tuning Only) | Exp2 (Pretraining Only) | Exp3 (Pretraining+FT) | NLI Exp1 | NLI Exp2 | NLI Exp3 |
|-------|-------------------------|-------------------------|------------------------|----------|----------|----------|
| LLaMA-3.1-8B | 0.4055 | 0.2193 | **0.4127** | -- | -- | -- |
| Mistral-7B | 0.3998 | 0.1639 | 0.3968 | -- | -- | -- |
| Qwen2.5-7B | 0.3582 | 0.2167 | 0.3609 | -- | -- | -- |
| Qwen2.5-1.5B | -- | 0.1249 | 0.3759 | -- | -- | -- |
| Phi-3-mini | 0.2782 | 0.1397 | 0.2951 | -- | -- | -- |
| **XLM-RoBERTa-Large** | **0.9921** | **0.9881** | **1.0000** | -- | -- | -- |

**Note:** LLM values = ROUGE-1 F1. XLM-RoBERTa = Accuracy (classification). NLI score for all models to be filled when computed in evaluation pipeline.

---

## Experiment 1: Fine-Tuning Only (Baseline)

### Language-Wise Performance (Exp1)

#### Table 1.1: ROUGE-1 F1 by Language

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.3541 | **0.4845** | 0.3823 | 0.4055 |
| Mistral-7B | **0.4299** | 0.3596 | **0.4077** | 0.3998 |
| Qwen2.5-7B | 0.3584 | 0.3612 | 0.3552 | 0.3582 |
| Phi-3-mini | 0.3622 | 0.1462 | 0.3188 | 0.2782 |
| Qwen2.5-1.5B | -- | -- | -- | -- |
| XLM-RoBERTa-Large | -- | -- | -- | -- |
| **Samples** | **331** | **311** | **326** | **968** |

**Note:** Only ROUGE-1 F1 has language-specific breakdowns. XLM-RoBERTa-Large is an encoder model; generation ROUGE breakdowns are N/A.

---

### Complexity-Wise Performance (Exp1)

#### Table 1.4: ROUGE-1 F1 by Complexity

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | **0.4363** | **0.3976** | **0.3832** | 0.4055 |
| Mistral-7B | **0.4227** | **0.3957** | **0.3815** | 0.3998 |
| Qwen2.5-7B | 0.3929 | 0.3582 | 0.3243 | 0.3582 |
| Phi-3-mini | 0.3190 | 0.2724 | 0.2439 | 0.2782 |
| Qwen2.5-1.5B | -- | -- | -- | -- |
| XLM-RoBERTa-Large | -- | -- | -- | -- |
| **Samples** | **318** | **326** | **324** | **968** |

**Note:** Only ROUGE-1 F1 has complexity-specific breakdowns. Overall ROUGE-L F1 and METEOR scores are provided in the main results table.

---

## Experiment 2: Pretraining Only (Zero-Shot)

### Language-Wise Performance (Exp2)

#### Table 2.1: ROUGE-1 F1 by Language

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.2386 | **0.2143** | **0.2046** | 0.2193 |
| Mistral-7B | **0.3159** | 0.0352 | 0.1323 | 0.1639 |
| Qwen2.5-7B | **0.3032** | 0.1884 | 0.1557 | 0.2167 |
| Qwen2.5-1.5B | 0.2596 | 0.0148 | 0.0932 | 0.1249 |
| Phi-3-mini | **0.2661** | 0.0462 | 0.1005 | 0.1397 |
| XLM-RoBERTa-Large | -- | -- | -- | -- |
| **Samples** | **331** | **311** | **326** | **968** |

**Key Finding:** English performs best across all models in zero-shot; Hindi is weakest for most models.

**Note:** Only ROUGE-1 F1 has language-specific breakdowns. Overall ROUGE-L F1 and METEOR scores are provided in the main results table.

---

### Complexity-Wise Performance (Exp2)

#### Table 2.4: ROUGE-1 F1 by Complexity

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | **0.2787** | **0.2193** | **0.1611** | 0.2193 |
| Mistral-7B | 0.1909 | 0.1599 | 0.1414 | 0.1639 |
| Qwen2.5-7B | **0.2517** | **0.2155** | **0.1834** | 0.2167 |
| Qwen2.5-1.5B | 0.1343 | 0.1239 | 0.1168 | 0.1249 |
| Phi-3-mini | 0.1809 | 0.1308 | 0.1082 | 0.1397 |
| XLM-RoBERTa-Large | -- | -- | -- | -- |
| **Samples** | **318** | **326** | **324** | **968** |

**Key Finding:** Professional complexity consistently outperforms Intermediate and Layman across all models.

**Note:** Only ROUGE-1 F1 has complexity-specific breakdowns. Overall ROUGE-L F1 and METEOR scores are provided in the main results table.

---

## Experiment 3: Pretraining + Fine-Tuning (Full Pipeline)

### Language-Wise Performance (Exp3)

#### Table 3.1: ROUGE-1 F1 by Language

| Model | English | Hindi | Code-Mixed | Avg |
|-------|---------|-------|------------|-----|
| LLaMA-3.1-8B | 0.3700 | **0.4911** | 0.3812 | 0.4127 |
| Mistral-7B | **0.4536** | 0.3324 | **0.4004** | 0.3968 |
| Qwen2.5-7B | 0.3621 | 0.3621 | 0.3585 | 0.3609 |
| Qwen2.5-1.5B | 0.3587 | **0.3950** | 0.3751 | 0.3759 |
| Phi-3-mini | **0.3809** | 0.1789 | 0.3189 | 0.2951 |
| XLM-RoBERTa-Large | -- | -- | -- | -- |
| **Samples** | **331** | **311** | **326** | **968** |

**Key Finding:** LLaMA-3.1-8B excels on Hindi (0.4911); Mistral-7B leads on English (0.4536).

**Note:** Only ROUGE-1 F1 has language-specific breakdowns. Overall ROUGE-L F1 and METEOR scores are provided in the main results table.

---

### Complexity-Wise Performance (Exp3)

#### Table 3.4: ROUGE-1 F1 by Complexity

| Model | Professional | Intermediate | Layman | Avg |
|-------|-------------|-------------|--------|-----|
| LLaMA-3.1-8B | **0.4489** | **0.4017** | **0.3880** | 0.4127 |
| Mistral-7B | **0.4266** | **0.3890** | **0.3753** | 0.3968 |
| Qwen2.5-7B | 0.3935 | 0.3538 | 0.3360 | 0.3609 |
| Qwen2.5-1.5B | 0.3889 | 0.3703 | 0.3688 | 0.3759 |
| Phi-3-mini | 0.3359 | 0.2938 | 0.2565 | 0.2951 |
| XLM-RoBERTa-Large | -- | -- | -- | -- |
| **Samples** | **318** | **326** | **324** | **968** |

**Key Finding:** Professional complexity consistently outperforms Intermediate and Layman; performance decreases with simpler language.

**Note:** Only ROUGE-1 F1 has complexity-specific breakdowns. Overall ROUGE-L F1 and METEOR scores are provided in the main results table.

---

## Cross-Experiment Language Analysis

### English Performance Across Experiments

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | Best |
|-------|-----------------|----------------------|--------------------|------|
| LLaMA-3.1-8B | 0.3541 | 0.2386 | 0.3700 | **Exp3** |
| Mistral-7B | **0.4299** | 0.3159 | **0.4536** | **Exp3** |
| Qwen2.5-7B | 0.3584 | 0.3032 | 0.3621 | **Exp3** |
| Qwen2.5-1.5B | -- | 0.2596 | 0.3587 | **Exp3** |
| Phi-3-mini | 0.3622 | 0.2661 | **0.3809** | **Exp3** |
| XLM-RoBERTa-Large | -- | -- | -- | -- |

### Hindi Performance Across Experiments

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | Best |
|-------|----------------|----------------------|--------------------|------|
| LLaMA-3.1-8B | **0.4845** | 0.2143 | **0.4911** | **Exp3** |
| Mistral-7B | 0.3596 | 0.0352 | 0.3324 | **Exp1** |
| Qwen2.5-7B | 0.3612 | 0.1884 | 0.3621 | **Exp3** |
| Qwen2.5-1.5B | -- | 0.0148 | **0.3950** | **Exp3** |
| Phi-3-mini | 0.1462 | 0.0462 | 0.1789 | **Exp3** |
| XLM-RoBERTa-Large | -- | -- | -- | -- |

**Key Finding:** Hindi is challenging in zero-shot (Exp2) but improves significantly with fine-tuning (Exp1/Exp3).

### Code-Mixed Performance Across Experiments

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | Best |
|-------|----------------|----------------------|--------------------|------|
| LLaMA-3.1-8B | 0.3823 | 0.2046 | 0.3812 | **Exp1** |
| Mistral-7B | **0.4077** | 0.1323 | **0.4004** | **Exp1** |
| Qwen2.5-7B | 0.3552 | 0.1557 | 0.3585 | **Exp3** |
| Qwen2.5-1.5B | -- | 0.0932 | 0.3751 | **Exp3** |
| Phi-3-mini | 0.3188 | 0.1005 | 0.3189 | **Exp3** |
| XLM-RoBERTa-Large | -- | -- | -- | -- |

---

## Cross-Experiment Complexity Analysis

### Professional Complexity Across Experiments

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | Best |
|-------|----------------|----------------------|--------------------|------|
| LLaMA-3.1-8B | 0.4363 | 0.2787 | **0.4489** | **Exp3** |
| Mistral-7B | 0.4227 | 0.1909 | **0.4266** | **Exp3** |
| Qwen2.5-7B | 0.3929 | 0.2517 | 0.3935 | **Exp3** |
| Qwen2.5-1.5B | -- | 0.1343 | 0.3889 | **Exp3** |
| Phi-3-mini | 0.3190 | 0.1809 | **0.3359** | **Exp3** |
| XLM-RoBERTa-Large | -- | -- | -- | -- |

### Intermediate Complexity Across Experiments

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | Best |
|-------|----------------|----------------------|--------------------|------|
| LLaMA-3.1-8B | 0.3976 | 0.2193 | **0.4017** | **Exp3** |
| Mistral-7B | 0.3957 | 0.1599 | **0.3890** | **Exp1** |
| Qwen2.5-7B | 0.3582 | 0.2155 | 0.3538 | **Exp1** |
| Qwen2.5-1.5B | -- | 0.1239 | **0.3703** | **Exp3** |
| Phi-3-mini | 0.2724 | 0.1308 | **0.2938** | **Exp3** |
| XLM-RoBERTa-Large | -- | -- | -- | -- |

### Layman Complexity Across Experiments

| Model | Exp1 (FT only) | Exp2 (Pretrain only) | Exp3 (Pretrain+FT) | Best |
|-------|----------------|----------------------|--------------------|------|
| LLaMA-3.1-8B | 0.3832 | 0.1611 | **0.3880** | **Exp3** |
| Mistral-7B | 0.3815 | 0.1414 | **0.3753** | **Exp1** |
| Qwen2.5-7B | 0.3243 | 0.1834 | 0.3360 | **Exp3** |
| Qwen2.5-1.5B | -- | 0.1168 | **0.3688** | **Exp3** |
| Phi-3-mini | 0.2439 | 0.1082 | **0.2565** | **Exp3** |
| XLM-RoBERTa-Large | -- | -- | -- | -- |

---

## Summary Insights

### Language Performance Patterns

1. **English:** Consistently strong across all experiments; Mistral-7B leads in Exp1/Exp3 (0.43/0.45).
2. **Hindi:** Strongest in Exp1/Exp3 (LLaMA-3.1-8B: 0.48/0.49); weak in zero-shot Exp2.
3. **Code-Mixed:** Moderate performance; Mistral-7B leads in Exp1 (0.41), LLaMA-3.1-8B in Exp3 (0.38).

### Complexity Performance Patterns

1. **Professional:** Consistently highest scores across all experiments (0.45+ for top models in Exp3).
2. **Intermediate:** Moderate scores, between Professional and Layman.
3. **Layman:** Lowest scores, indicating simpler language is harder to generate accurately.

### Model Strengths

- **LLaMA-3.1-8B:** Best on Hindi (0.49) and Professional complexity (0.45).
- **Mistral-7B:** Best on English (0.45) and Code-Mixed (0.40).
- **Qwen2.5-7B:** Balanced across languages; consistent performance.
- **Qwen2.5-1.5B:** Strong recovery in Exp3 (0.38 ROUGE-1) despite Exp1 issues.
- **Phi-3-mini:** Best on English (0.38); struggles with Hindi (0.18).
- **XLM-RoBERTa-Large:** Encoder (classification): Exp1 Accuracy 99.21%, Exp2 98.81%, Exp3 100%. No per-language/per-complexity breakdown; no generation ROUGE.

### NLI Score (for all models)

**NLI** (Natural Language Inference / entailment-based consistency) is not yet computed in the evaluation pipeline. The **NLI Exp1 / NLI Exp2 / NLI Exp3** columns in the "All Final Results" and "Overall Performance and NLIC" tables are reserved for all models (LLMs and XLM-RoBERTa); currently placeholders "--". Once NLI is added (e.g. in `evaluation/metrics.py` and in `evaluate_generation.py` for generation models and in `evaluate_and_save.py` or equivalent for XLM-R), fill in the NLI columns for every model and experiment.
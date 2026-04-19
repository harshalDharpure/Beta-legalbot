# Model Comparison Tables (Exp1)

**Generated:** February 07, 2026

## Table 1: Overall Performance Metrics

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | METEOR |
|-------|--------|--------|--------|--------|------------|------------|------------|--------|
| LLaMA-3.1-8B | 0.2660 | 0.1375 | 0.0791 | 0.0451 | 0.4055 | 0.1381 | 0.2775 | 0.2702 |
| Mistral-7B | 0.2542 | 0.1290 | 0.0745 | 0.0430 | 0.3998 | 0.1300 | 0.2639 | 0.2386 |
| Qwen2.5-7B | 0.2103 | 0.0993 | 0.0537 | 0.0296 | 0.3582 | 0.1069 | 0.2334 | 0.2268 |
| Qwen2.5-1.5B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Phi-3-mini | 0.1855 | 0.0853 | 0.0436 | 0.0232 | 0.2782 | 0.0821 | 0.1711 | 0.1852 |

## Table 2: Language-Specific Performance (ROUGE-1 F1)

| Language | LLaMA-3.1-8B | Mistral-7B | Qwen2.5-7B | Qwen2.5-1.5B | Phi-3-mini | Samples |
|----------|--------|--------|--------|--------|--------|---------|
| English | 0.3541 | 0.4299 | 0.3584 | – | – | 331 |
| Hindi | 0.4845 | 0.3596 | 0.3612 | – | – | 311 |
| Code-Mixed | 0.3823 | 0.4077 | 0.3552 | – | – | 326 |

## Table 3: Complexity-Specific Performance (ROUGE-1 F1)

| Complexity | LLaMA-3.1-8B | Mistral-7B | Qwen2.5-7B | Qwen2.5-1.5B | Phi-3-mini | Samples |
|------------|--------|--------|--------|--------|--------|---------|
| Professional | 0.4363 | 0.4227 | 0.3929 | – | – | 318 |
| Intermediate | 0.3976 | 0.3957 | 0.3582 | – | – | 326 |
| Layman | 0.3832 | 0.3815 | 0.3243 | – | – | 324 |

## Table 4: Response Length Comparison

| Model | Avg Reference Length | Avg Candidate Length | Ratio | Difference |
|-------|----------------------|----------------------|-------|------------|
| LLaMA-3.1-8B | 89.92 | 152.76 | 1.70 | +62.84 |
| Mistral-7B | 89.92 | 102.66 | 1.14 | +12.74 |
| Qwen2.5-7B | 89.92 | 135.64 | 1.51 | +45.72 |
| Qwen2.5-1.5B | 89.92 | 1.00 | 0.01 | -88.92 |
| Phi-3-mini | 89.92 | 115.91 | 1.29 | +25.99 |

## Table 5: Model Ranking Summary (ROUGE-1 F1)

| Rank | Model | ROUGE-1 F1 | Samples |
|------|-------|------------|---------|
| 1 | LLaMA-3.1-8B | 0.4055 | 968 |
| 2 | Mistral-7B | 0.3998 | 968 |
| 3 | Qwen2.5-7B | 0.3582 | 968 |
| 4 | Phi-3-mini | 0.2782 | 968 |
| 5 | Qwen2.5-1.5B | 0.0000 | 968 |
# Model Comparison Tables (Exp1)

**Generated:** February 06, 2026

## Table 1: Overall Performance Metrics

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | METEOR |
|-------|--------|--------|--------|--------|------------|------------|------------|--------|
| LLaMA-3.1-8B | 0.2660 | 0.1375 | 0.0791 | 0.0451 | 0.4055 | 0.1381 | 0.2775 | 0.2702 |
| Mistral-7B | 0.2542 | 0.1290 | 0.0745 | 0.0430 | 0.3998 | 0.1300 | 0.2639 | 0.2386 |
| Qwen2.5-7B | 0.2103 | 0.0993 | 0.0537 | 0.0296 | 0.3582 | 0.1069 | 0.2334 | 0.2268 |
| Qwen2.5-1.5B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Table 2: Language-Specific Performance (ROUGE-1 F1)

| Language | LLaMA-3.1-8B | Mistral-7B | Qwen2.5-7B | Qwen2.5-1.5B | Samples |
|----------|--------|--------|--------|--------|---------|
| English | – | – | – | – | 0 |
| Hindi | – | – | – | – | 0 |
| Code-Mixed | – | – | – | – | 0 |

## Table 3: Complexity-Specific Performance (ROUGE-1 F1)

| Complexity | LLaMA-3.1-8B | Mistral-7B | Qwen2.5-7B | Qwen2.5-1.5B | Samples |
|------------|--------|--------|--------|--------|---------|
| Professional | – | – | – | – | 0 |
| Intermediate | – | – | – | – | 0 |
| Layman | – | – | – | – | 0 |

## Table 4: Response Length Comparison

| Model | Avg Reference Length | Avg Candidate Length | Ratio | Difference |
|-------|----------------------|----------------------|-------|------------|
| LLaMA-3.1-8B | 89.92 | 152.76 | 1.70 | +62.84 |
| Mistral-7B | 89.92 | 102.66 | 1.14 | +12.74 |
| Qwen2.5-7B | 89.92 | 135.64 | 1.51 | +45.72 |
| Qwen2.5-1.5B | 89.92 | 1.00 | 0.01 | -88.92 |

## Table 5: Model Ranking Summary (ROUGE-1 F1)

| Rank | Model | ROUGE-1 F1 | Samples |
|------|-------|------------|---------|
| 1 | LLaMA-3.1-8B | 0.4055 | 968 |
| 2 | Mistral-7B | 0.3998 | 968 |
| 3 | Qwen2.5-7B | 0.3582 | 968 |
| 4 | Qwen2.5-1.5B | 0.0000 | 968 |
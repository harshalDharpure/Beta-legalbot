# POCSO Dataset - Experimental Setup

Professional experimental framework for multilingual POCSO legal dialogue research.

## Structure

```
experiments/
├── README.md                          # This file
├── exp1_supervised_baseline/         # Experiment 1: Supervised 80/20
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── train.jsonl
│       ├── test.jsonl
│       └── combined.jsonl
├── exp2_monolingual_baseline/        # Experiment 2: Monolingual
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── hindi_train.jsonl
│       ├── hindi_test.jsonl
│       ├── code_mixed_train.jsonl
│       ├── code_mixed_test.jsonl
│       ├── english_train.jsonl
│       └── english_test.jsonl
├── exp3_zeroshot_transfer/          # Experiment 3: Zero-shot
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── hindi_code_mixed_to_english/
│       ├── english_code_mixed_to_hindi/
│       └── hindi_english_to_code_mixed/
├── exp4_fewshot_learning/           # Experiment 4: Few-shot
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── few5/
│       ├── few10/
│       ├── few20/
│       └── few50/
├── exp5_comparison/                  # Experiment 5: Comparison tools
│   └── (analysis scripts)
└── common/                           # Common resources
    ├── load_combined.py
    └── combined_dataset.jsonl
```

## Quick Start

### Experiment 1: Supervised Baseline
```python
import sys
sys.path.append('experiments/exp1_supervised_baseline')
from load_data import load_train_test

train, test = load_train_test()
```

### Experiment 2: Monolingual
```python
import sys
sys.path.append('experiments/exp2_monolingual_baseline')
from load_data import load_language_data

hindi_train, hindi_test = load_language_data('hindi')
```

### Experiment 3: Zero-Shot
```python
import sys
sys.path.append('experiments/exp3_zeroshot_transfer')
from load_data import load_zeroshot_config

train, test = load_zeroshot_config('hindi_code_mixed_to_english')
```

### Experiment 4: Few-Shot
```python
import sys
sys.path.append('experiments/exp4_fewshot_learning')
from load_data import load_fewshot_config

train, test = load_fewshot_config(few_size=10, direction='hindi_code_mixed_to_english')
```

## Experiments Overview

| Experiment | Description | Train Size | Test Size |
|------------|-------------|------------|-----------|
| **Exp1** | Supervised Baseline (Stratified) | 948 | 252 |
| **Exp2** | Monolingual (per language) | 316 | 84 |
| **Exp3** | Zero-Shot Cross-Lingual | 800 | 400 |
| **Exp4** | Few-Shot (5,10,20,50 examples) | 800+N | 400-N |

## Recommended Workflow

1. **Start with Exp1**: Establish supervised baseline (upper bound)
2. **Compare with Exp2**: Understand monolingual vs multilingual
3. **Evaluate Exp3**: Test zero-shot transfer capabilities
4. **Analyze Exp4**: Find optimal few-shot strategies

## Notes

- All experiments use random seed 42 for reproducibility
- Stratified splits maintain perfect distribution
- Each experiment has its own README and helper scripts
- Clean, professional structure for research

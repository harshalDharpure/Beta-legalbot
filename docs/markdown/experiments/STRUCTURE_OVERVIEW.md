# Professional Experimental Structure - Complete Overview

## ✅ Structure Created

```
experiments/
├── README.md                          # Main documentation
├── STRUCTURE_OVERVIEW.md              # This file
│
├── exp1_supervised_baseline/         # Experiment 1
│   ├── README.md                      # Experiment documentation
│   ├── load_data.py                   # Data loading helper
│   └── data/
│       ├── train.jsonl                # 948 entries
│       ├── test.jsonl                 # 252 entries
│       └── combined.jsonl             # 1,200 entries
│
├── exp2_monolingual_baseline/        # Experiment 2
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── hindi_train.jsonl          # 316 entries
│       ├── hindi_test.jsonl           # 84 entries
│       ├── code_mixed_train.jsonl     # 316 entries
│       ├── code_mixed_test.jsonl     # 84 entries
│       ├── english_train.jsonl        # 316 entries
│       └── english_test.jsonl         # 84 entries
│
├── exp3_zeroshot_transfer/          # Experiment 3
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── hindi_code_mixed_to_english/
│       │   ├── train.jsonl            # 800 entries
│       │   └── test.jsonl             # 400 entries
│       ├── english_code_mixed_to_hindi/
│       │   ├── train.jsonl            # 800 entries
│       │   └── test.jsonl             # 400 entries
│       └── hindi_english_to_code_mixed/
│           ├── train.jsonl            # 800 entries
│           └── test.jsonl             # 400 entries
│
├── exp4_fewshot_learning/           # Experiment 4
│   ├── README.md
│   ├── load_data.py
│   └── data/
│       ├── few5/
│       │   ├── hindi_code_mixed_to_english/
│       │   └── english_code_mixed_to_hindi/
│       ├── few10/
│       │   ├── hindi_code_mixed_to_english/
│       │   └── english_code_mixed_to_hindi/
│       ├── few20/
│       │   ├── hindi_code_mixed_to_english/
│       │   └── english_code_mixed_to_hindi/
│       └── few50/
│           ├── hindi_code_mixed_to_english/
│           └── english_code_mixed_to_hindi/
│
├── exp5_comparison/                  # Experiment 5 (for analysis)
│   └── (analysis scripts can go here)
│
└── common/                           # Common resources
    ├── load_combined.py
    └── combined_dataset.jsonl        # 1,200 entries
```

## 📊 Statistics

- **Total JSONL files**: 32
- **Total experiments**: 4 main + 1 comparison
- **Each experiment has**:
  - ✅ README.md (complete documentation)
  - ✅ load_data.py (helper script)
  - ✅ data/ directory (organized files)

## 🚀 Quick Usage Examples

### Experiment 1: Supervised Baseline
```python
import sys
sys.path.append('experiments/exp1_supervised_baseline')
from load_data import load_train_test

train, test = load_train_test()
print(f"Train: {len(train)}, Test: {len(test)}")
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

## ✨ Features

1. **Clean Organization**: Each experiment in its own folder
2. **Self-Contained**: Each experiment has its own README and loader
3. **Professional Structure**: Follows research best practices
4. **Easy to Use**: Simple import and load functions
5. **Well Documented**: README in each experiment folder
6. **Reproducible**: All splits use random seed 42

## 📝 Notes

- All experiments are independent and self-contained
- Each experiment can be used independently
- Helper scripts make data loading simple
- Structure is publication-ready
- Easy to extend with new experiments

---

**Status**: ✅ **PROFESSIONAL STRUCTURE COMPLETE**

# Experiment 2: Monolingual Baselines

## Overview
Language-specific baselines to compare monolingual vs multilingual performance.

## Datasets

### Hindi
- **Train**: 316 entries
- **Test**: 84 entries

### Code-mixed
- **Train**: 316 entries
- **Test**: 84 entries

### English
- **Train**: 316 entries
- **Test**: 84 entries

## Purpose
- Compare monolingual vs multilingual performance
- Understand language-specific challenges
- Baseline for cross-lingual transfer experiments

## Usage

```python
from load_data import load_language_data

# Load Hindi data
hindi_train, hindi_test = load_language_data('hindi')

# Load Code-mixed data
cm_train, cm_test = load_language_data('code_mixed')

# Load English data
en_train, en_test = load_language_data('english')
```

## Files
- `data/hindi_train.jsonl`, `data/hindi_test.jsonl`
- `data/code_mixed_train.jsonl`, `data/code_mixed_test.jsonl`
- `data/english_train.jsonl`, `data/english_test.jsonl`

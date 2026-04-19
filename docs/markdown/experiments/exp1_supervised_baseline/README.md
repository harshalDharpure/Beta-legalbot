# Experiment 1: Supervised Baseline (Stratified 80/20)

## Overview
Standard supervised learning baseline with stratified train/test split maintaining perfect distribution across languages, complexity levels, and turn count buckets.

## Dataset
- **Train**: 948 entries (79%)
- **Test**: 252 entries (21%)
- **Total**: 1,200 entries

## Distribution
- **Languages**: 33.3% Hindi, 33.3% Code-mixed, 33.3% English
- **Complexity**: ~33.3% each (Layman, Intermediate, Professional)
- **Buckets**: A(25%), B(26.1%), C(26.8%), D(22.2%)

## Purpose
- Establish upper bound performance
- Test multilingual learning from mixed data
- Baseline for comparison with other experiments

## Usage

```python
from load_data import load_train_test

train_data, test_data = load_train_test()
print(f"Train: {len(train_data)}, Test: {len(test_data)}")
```

## Files
- `data/train.jsonl` - Training set
- `data/test.jsonl` - Test set
- `data/combined.jsonl` - Full dataset

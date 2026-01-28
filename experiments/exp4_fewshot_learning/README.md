# Experiment 4: Few-Shot Cross-Lingual Learning

## Overview
Test few-shot learning with minimal target language examples.

## Few-Shot Sizes
- **5 examples**: Minimal few-shot
- **10 examples**: Small few-shot
- **20 examples**: Medium few-shot
- **50 examples**: Large few-shot

## Configurations

For each few-shot size, two directions:

### 1. Hindi + Code-mixed + N English → English
- **Train**: 800 + N entries (H+CM+N EN examples)
- **Test**: 400 - N entries (remaining English)
- **Directory**: `data/few{N}/hindi_code_mixed_to_english/`

### 2. English + Code-mixed + N Hindi → Hindi
- **Train**: 800 + N entries (EN+CM+N HI examples)
- **Test**: 400 - N entries (remaining Hindi)
- **Directory**: `data/few{N}/english_code_mixed_to_hindi/`

## Purpose
- Evaluate sample efficiency
- Find optimal few-shot strategies
- Compare with zero-shot and supervised baselines

## Usage

```python
from load_data import load_fewshot_config

# Load few-shot with 10 examples: H+CM+10EN → EN
train, test = load_fewshot_config(few_size=10, direction='hindi_code_mixed_to_english')

# Load few-shot with 20 examples: EN+CM+20HI → HI
train, test = load_fewshot_config(few_size=20, direction='english_code_mixed_to_hindi')
```

## Files
Each configuration has:
- `data/few{N}/{direction}/train.jsonl` - Training set
- `data/few{N}/{direction}/test.jsonl` - Test set

## Expected Results
Performance should increase with more few-shot examples, approaching supervised baseline.

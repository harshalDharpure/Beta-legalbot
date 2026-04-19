# Experiment 3: Zero-Shot Cross-Lingual Transfer

## Overview
Test cross-lingual transfer without any target language training data.

## Configurations

### 1. Hindi + Code-mixed → English
- **Train**: 800 entries (Hindi + Code-mixed)
- **Test**: 400 entries (English)
- **Directory**: `data/hindi_code_mixed_to_english/`

### 2. English + Code-mixed → Hindi
- **Train**: 800 entries (English + Code-mixed)
- **Test**: 400 entries (Hindi)
- **Directory**: `data/english_code_mixed_to_hindi/`

### 3. Hindi + English → Code-mixed
- **Train**: 800 entries (Hindi + English)
- **Test**: 400 entries (Code-mixed)
- **Directory**: `data/hindi_english_to_code_mixed/`

## Purpose
- Evaluate cross-lingual transfer capabilities
- Test model generalization across languages
- Compare with supervised baseline

## Usage

```python
from load_data import load_zeroshot_config

# Load Hindi+Code-mixed → English
train, test = load_zeroshot_config('hindi_code_mixed_to_english')

# Load English+Code-mixed → Hindi
train, test = load_zeroshot_config('english_code_mixed_to_hindi')

# Load Hindi+English → Code-mixed
train, test = load_zeroshot_config('hindi_english_to_code_mixed')
```

## Files
Each configuration has:
- `data/{config}/train.jsonl` - Training set
- `data/{config}/test.jsonl` - Test set

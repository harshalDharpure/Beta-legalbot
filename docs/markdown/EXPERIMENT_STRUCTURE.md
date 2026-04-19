# 🎯 Experiment Structure - Clear Overview

## Experiment Design

### **Exp1: Finetuning Only** (Baseline)
- **What**: Direct finetuning on dialogue data
- **No**: Pretraining
- **Yes**: Finetuning on 70/10/20 split
- **Purpose**: Baseline performance
- **Expected**: Good performance (baseline)

---

### **Exp2: Pretraining Only**
- **What**: Pretrain on legal corpus, then evaluate zero-shot
- **Yes**: Pretraining on legal corpus
- **No**: Finetuning on dialogue data
- **Purpose**: Test pretraining effectiveness alone
- **Expected**: Lower performance (no task-specific training)

---

### **Exp3: Pretraining + Finetuning** (Full Pipeline)
- **What**: Pretrain on legal corpus, then finetune on dialogue data
- **Yes**: Pretraining on legal corpus
- **Yes**: Finetuning on 70/10/20 split
- **Purpose**: Best performance (full pipeline)
- **Expected**: **Best performance** (domain + task adaptation)

---

### **Exp4: Zero-Shot Transfer**
- **What**: Cross-lingual generation
- **Train**: 2 languages (e.g., Hindi + CodeMixed)
- **Test**: 3rd language (e.g., English)
- **Purpose**: Test cross-lingual transfer
- **Expected**: Moderate performance

---

### **Exp5: Few-Shot Learning**
- **What**: Minimal data fine-tuning
- **Train**: 5, 10, 20, 50 examples
- **Test**: Full test set
- **Purpose**: Test efficiency with minimal data
- **Expected**: Lower than Exp1, but efficient

---

## 📊 Comparison Matrix

| Experiment | Pretraining | Finetuning | Expected Performance |
|------------|-------------|------------|---------------------|
| **Exp1** | ❌ No | ✅ Yes | Baseline (Good) |
| **Exp2** | ✅ Yes | ❌ No | Lower (No task training) |
| **Exp3** | ✅ Yes | ✅ Yes | **Best** (Full pipeline) |
| **Exp4** | - | Cross-lingual | Moderate |
| **Exp5** | - | Few-shot | Lower (Minimal data) |

---

## 🔬 Ablation Study

**Research Question**: Does pretraining help?

**Comparison:**
1. **Exp3 vs Exp1**: Improvement from pretraining
   - Exp3 (Pretraining + Finetuning) vs Exp1 (Finetuning only)
   - **Hypothesis**: Exp3 > Exp1 (pretraining helps)

2. **Exp3 vs Exp2**: Improvement from finetuning
   - Exp3 (Pretraining + Finetuning) vs Exp2 (Pretraining only)
   - **Hypothesis**: Exp3 > Exp2 (finetuning helps)

3. **Exp1 vs Exp2**: Finetuning vs Pretraining alone
   - Exp1 (Finetuning only) vs Exp2 (Pretraining only)
   - **Hypothesis**: Exp1 > Exp2 (task-specific training > domain pretraining alone)

---

## 📁 Folder Structure

```
experiments/
├── exp1_finetuning_only/        # Exp1: Baseline
│   ├── train.jsonl (840)
│   ├── val.jsonl (120)
│   └── test.jsonl (240)
│
├── exp2_pretraining_only/        # Exp2: Pretraining only
│   ├── pretraining/
│   │   └── legal_corpus/
│   └── evaluation/
│       └── test.jsonl (240)     # Zero-shot evaluation
│
├── exp3_pretraining_finetuning/ # Exp3: Full pipeline
│   ├── pretraining/
│   │   └── legal_corpus/
│   └── finetuning/
│       ├── train.jsonl (840)
│       ├── val.jsonl (120)
│       └── test.jsonl (240)
│
├── exp4_zeroshot_transfer/      # Exp4: Cross-lingual
│   ├── hindi_codemixed_to_english/
│   ├── english_codemixed_to_hindi/
│   └── hindi_english_to_codemixed/
│
└── exp5_fewshot_learning/       # Exp5: Few-shot
    ├── few5/
    ├── few10/
    ├── few20/
    └── few50/
```

---

## 🎯 Expected Results Ranking

**Performance Ranking (Expected):**
1. **Exp3** (Pretraining + Finetuning) - **Best** ⭐
2. **Exp1** (Finetuning only) - Good (baseline)
3. **Exp4** (Zero-shot) - Moderate
4. **Exp5** (Few-shot) - Lower (minimal data)
5. **Exp2** (Pretraining only) - Lowest (no task training)

---

## ✅ Key Points

1. **Exp1 = Baseline**: Direct finetuning, no pretraining
2. **Exp2 = Pretraining Only**: Test pretraining effectiveness alone
3. **Exp3 = Full Pipeline**: Pretraining + Finetuning (expected best)
4. **Ablation Study**: Compare Exp1, Exp2, Exp3 to understand contributions
5. **Exp4 & Exp5**: Additional experiments for transfer and efficiency

---

**Status**: ✅ **CLEAR STRUCTURE**  
**Ready to implement**: Yes

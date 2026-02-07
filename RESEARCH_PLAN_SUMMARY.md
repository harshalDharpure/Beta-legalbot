# 🎯 Research Plan Summary - Quick Reference

## Current Status ✅

**Completed:**
- Classification task (complexity prediction)
- 80/20 train/test split
- 2 models trained (XLM-RoBERTa, MuRIL)
- Zero-shot and few-shot experiments (classification)

**Missing (Critical):**
- ❌ Generation task (response generation)
- ❌ Validation set (70/10/20 split)
- ❌ Generation models (LLaMA, Mistral, Qwen, Phi-3)
- ❌ Generation metrics (BLEU, ROUGE, METEOR)
- ❌ Pretraining experiments

---

## 🎯 Research Plan (4 Phases)

### Phase 1: Dataset Preparation
**Goal**: Create 70/10/20 train/val/test split

**Actions:**
1. Load all 1,200 samples
2. Stratified split (language, complexity, bucket)
3. Create train (840), val (120), test (240) files
4. Convert to generation format (user query → assistant response)

**Output**: `data/splits/train_70.jsonl`, `val_10.jsonl`, `test_20.jsonl`

---

### Phase 2: Model Setup
**Goal**: Setup 7 generation models

**Models:**
1. **LLaMA-3.1-8B** (QLoRA)
2. **Mistral-7B** (QLoRA)
3. **Qwen2.5-7B** (QLoRA)
4. **Qwen2.5-1.5B** (Full fine-tuning)
5. **Phi-3-mini** (Full fine-tuning)
6. **XLM-RoBERTa-Large** (Retrain for generation)
7. **MuRIL-Large** (Retrain for generation)

**Output**: Model folders with configs and training scripts

---

### Phase 3: Experiments

#### Exp1: Supervised Baseline (70/10/20)
- Train all 7 models
- Evaluate on test set
- Generate Table 1 (Overall Performance)

#### Exp2: Pretraining + Finetuning
- **Exp2**: Pretrain on legal corpus (LLaMA, Mistral, Qwen2.5-7B)
- **Exp1**: Finetune pretrained models on dialogue data
- Compare: Pretraining+Finetuning vs Finetuning only

#### Exp3: Zero-Shot Transfer
- Train on 2 languages → Test on 3rd
- Generate Table 2 (Language-Specific Performance)

#### Exp4: Few-Shot Learning
- Train with 5, 10, 20, 50 examples
- Analyze performance vs shot size

---

### Phase 4: Evaluation & Tables

**Metrics to Calculate:**
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1
- METEOR
- Response Length (avg reference, avg candidate, ratio, difference)

**Tables to Generate:**
1. **Table 1**: Overall Performance Metrics (all models, all metrics)
2. **Table 2**: Language-Specific Performance (ROUGE-1 F1 by language)
3. **Table 3**: Complexity-Specific Performance (ROUGE-1 F1 by complexity)
4. **Table 4**: Response Length Comparison
5. **Table 5**: Model Ranking Summary (ROUGE-1 F1)

---

## 📊 Expected Results (Based on Image)

**Best Model (LLaMA-3.1-8B):**
- BLEU-1: ~0.32
- BLEU-4: ~0.05
- ROUGE-1 F1: ~0.36
- ROUGE-L F1: ~0.23
- METEOR: ~0.37

**Language Performance (ROUGE-1 F1):**
- English: ~0.42
- Hindi: ~0.32
- Code-mixed: ~0.33

---

## 🚀 Next Steps (Priority Order)

1. **✅ Create 70/10/20 split** (Day 1)
2. **✅ Setup generation models** (Day 2-3)
3. **✅ Train Exp1 (Supervised Baseline)** (Day 4-8)
4. **✅ Train Exp2 (Pretraining + Finetuning)** (Day 9-12)
5. **✅ Train Exp3 (Zero-Shot)** (Day 13-14)
6. **✅ Train Exp4 (Few-Shot)** (Day 15)
7. **✅ Evaluate all models** (Day 16-17)
8. **✅ Generate tables** (Day 18)
9. **✅ Documentation** (Day 19-20)

---

## 📁 Repository Structure

```
legal-bot/
├── data/splits/              # 70/10/20 split
├── experiments/
│   ├── exp1_supervised/      # 70/10/20 baseline
│   ├── exp2_pretraining/     # Pretraining + finetuning
│   ├── exp3_zeroshot/        # Zero-shot transfer
│   └── exp4_fewshot/         # Few-shot learning
├── models/
│   ├── llama3.1_8b/          # Each model in separate folder
│   ├── mistral_7b/
│   ├── qwen2.5_7b/
│   ├── qwen2.5_1.5b/
│   ├── phi3_mini/
│   ├── xlmr_large/           # Retrain for generation
│   └── muril_large/          # Retrain for generation
└── results/
    └── tables/               # Paper-ready tables
```

---

## ⚠️ Critical Notes

1. **Task Shift**: Classification → Generation
   - Need to convert dialogue pairs to input/output format
   - Format: `"User: {query}\nAssistant: {response}"`

2. **Test Set**: Hide complete dialogue pairs
   - No overlap between train/val/test
   - Use validation set for hyperparameter tuning
   - Test set only for final evaluation

3. **GPU Constraints**: Use QLoRA for large models
   - LLaMA-3.1-8B: QLoRA (4-bit)
   - Mistral-7B: QLoRA (4-bit)
   - Qwen2.5-7B: QLoRA (4-bit)
   - Small models: Full fine-tuning

4. **Reproducibility**: Random seed 42, save all checkpoints

---

## 📝 Paper Title Suggestion

**"Multilingual Legal Dialogue Generation: Zero-Shot Learning for POCSO Act Queries"**

**Alternative:**
**"Cross-Lingual Legal Response Generation: A Comprehensive Study on POCSO Act Dialogues"**

---

**Status**: 🟢 Ready to implement  
**Start**: Create 70/10/20 dataset split

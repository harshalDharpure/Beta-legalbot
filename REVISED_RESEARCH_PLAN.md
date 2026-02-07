# 🎯 Revised Research Plan: Top-Tier A* Paper
## Multilingual Legal Dialogue Generation for POCSO Act

**Research Scientist Perspective**: Google Principal Research Scientist / World's Best NLP Master

---

## 📊 Executive Summary

### Current State Analysis

**✅ Completed:**
1. **Classification Task**: Complexity classification (layman/intermediate/professional)
   - Models: XLM-RoBERTa-Large (95.24%), MuRIL-Large (72.62%)
   - Experiments: Supervised baseline, monolingual, zero-shot, few-shot
   - Dataset: 1,200 dialogues (400 per language: Hindi, English, Code-mixed)
   - Current Split: 80/20 train/test

**❌ Missing (Critical for A* Paper):**
1. **Generation Task**: Response generation (user query → legal response)
2. **Validation Set**: No validation split (only 80/20 train/test)
3. **Generation Models**: No seq2seq models trained
4. **Pretraining Experiments**: No domain-specific pretraining
5. **Modern LLMs**: No evaluation of LLaMA-3.1, Mistral, Qwen2.5, Phi-3
6. **Generation Metrics**: No BLEU, ROUGE, METEOR scores

---

## 🎯 Revised Experimental Design

### Experiment 1: Finetuning Only (Baseline)

**Purpose**: Establish baseline with direct finetuning on dialogue data

**Setup:**
- **Data**: 70/10/20 train/val/test split
- **Train**: 840 samples (70%)
- **Val**: 120 samples (10%)
- **Test**: 240 samples (20%)
- **Task**: Response generation (user query → assistant response)
- **Training**: Direct finetuning on dialogue pairs (no pretraining)
- **Models**: All 7 models
- **Metrics**: BLEU-1/2/3/4, ROUGE-1/2/L F1, METEOR, BERTScore

**What This Tests:**
- Baseline performance without domain pretraining
- Direct adaptation to dialogue generation task
- Model comparison on same data

**Expected Output:**
- Baseline performance for all models
- Best model identification
- Response quality assessment

---

### Experiment 2: Pretraining Only

**Purpose**: Evaluate domain-specific pretraining without task finetuning

**Phase 2.1: Pretraining**
- **Data**: Legal corpus (POCSO Act text, legal documents, case summaries)
- **Task**: Masked Language Modeling (MLM) for encoder models OR Causal Language Modeling (CLM) for decoder models
- **Models**: LLaMA-3.1-8B, Mistral-7B, Qwen2.5-7B, Qwen2.5-1.5B, Phi-3-mini
- **Duration**: 1-2 epochs on legal corpus
- **Output**: Pretrained checkpoints

**Phase 2.2: Direct Evaluation (No Finetuning)**
- **Data**: Test set (240 samples)
- **Task**: Zero-shot generation (no finetuning on dialogue data)
- **Models**: Pretrained checkpoints from Phase 2.1
- **Prompt**: Use instruction templates for generation
- **Output**: Generation performance without task-specific finetuning

**What This Tests:**
- Effectiveness of domain pretraining alone
- Zero-shot generation capability after pretraining
- Whether pretraining alone is sufficient

**Expected Output:**
- Pretraining-only performance
- Comparison with Exp1 (finetuning only)
- Insight into pretraining contribution

---

### Experiment 3: Pretraining + Finetuning (Full Pipeline)

**Purpose**: Complete pipeline with both pretraining and task finetuning

**Phase 3.1: Pretraining (Same as Exp2)**
- **Data**: Legal corpus
- **Task**: MLM/CLM pretraining
- **Models**: LLaMA-3.1-8B, Mistral-7B, Qwen2.5-7B, Qwen2.5-1.5B, Phi-3-mini
- **Output**: Pretrained checkpoints

**Phase 3.2: Finetuning (Same as Exp1)**
- **Data**: 70/10/20 train/val/test split
- **Train**: 840 samples (70%)
- **Val**: 120 samples (10%)
- **Test**: 240 samples (20%)
- **Task**: Response generation
- **Models**: Pretrained checkpoints from Phase 3.1
- **Output**: Final fine-tuned models

**What This Tests:**
- Full pipeline effectiveness
- Whether pretraining + finetuning > finetuning only
- Optimal training strategy

**Expected Output:**
- Best performance (expected)
- Comparison with Exp1 and Exp2
- Ablation study results

---

### Experiment 4: Zero-Shot Transfer

**Purpose**: Cross-lingual generation without training on target language

**Scenarios:**
1. **Train on Hindi+CodeMixed → Test on English**
2. **Train on English+CodeMixed → Test on Hindi**
3. **Train on Hindi+English → Test on CodeMixed**

**Setup:**
- **Train**: 560 samples (2 languages × 280 each)
- **Val**: 80 samples (2 languages × 40 each)
- **Test**: 80 samples (target language)
- **Models**: Best models from Exp1, Exp2, Exp3
- **Metrics**: Language-specific ROUGE-1 F1, BLEU scores

**What This Tests:**
- Cross-lingual transfer capability
- Best transfer direction
- Language-specific performance

**Expected Output:**
- Zero-shot transfer performance
- Best transfer direction identification
- Language-specific analysis

---

### Experiment 5: Few-Shot Learning

**Purpose**: Minimal data fine-tuning

**Shot Sizes**: 5, 10, 20, 50 examples per language

**Setup:**
- **Train**: N samples (few-shot size)
- **Val**: 20 samples
- **Test**: 240 samples (full test set)
- **Models**: Best models from Exp1, Exp2, Exp3
- **Metrics**: Performance vs shot size curve

**What This Tests:**
- Optimal few-shot size
- Model efficiency for few-shot
- Minimal data scenarios

**Expected Output:**
- Optimal few-shot size
- Model comparison for few-shot
- Efficiency analysis

---

## 📊 Experiment Comparison Matrix

| Experiment | Pretraining | Finetuning | Purpose | Expected Performance |
|------------|-------------|------------|---------|---------------------|
| **Exp1** | ❌ No | ✅ Yes | Baseline | Baseline |
| **Exp2** | ✅ Yes | ❌ No | Pretraining only | Lower than Exp1 |
| **Exp3** | ✅ Yes | ✅ Yes | Full pipeline | **Best** |
| **Exp4** | - | Cross-lingual | Zero-shot transfer | Moderate |
| **Exp5** | - | Few-shot | Minimal data | Lower than Exp1 |

---

## 🗂️ Repository Structure (Revised)

```
legal-bot/
├── data/
│   ├── splits/
│   │   ├── train_70.jsonl          # 840 samples
│   │   ├── val_10.jsonl             # 120 samples
│   │   └── test_20.jsonl            # 240 samples
│   └── pretraining/
│       └── legal_corpus/            # Legal text for pretraining
│           ├── pocso_act.txt
│           ├── legal_documents/
│           └── case_summaries/
│
├── experiments/
│   ├── exp1_finetuning_only/        # Exp1: Finetuning baseline
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── exp2_pretraining_only/       # Exp2: Pretraining only
│   │   ├── pretraining/
│   │   │   └── legal_corpus/
│   │   └── evaluation/
│   │       └── test.jsonl            # Zero-shot evaluation
│   ├── exp3_pretraining_finetuning/ # Exp3: Full pipeline
│   │   ├── pretraining/
│   │   │   └── legal_corpus/
│   │   └── finetuning/
│   │       ├── train.jsonl
│   │       ├── val.jsonl
│   │       └── test.jsonl
│   ├── exp4_zeroshot_transfer/      # Exp4: Zero-shot
│   │   ├── hindi_codemixed_to_english/
│   │   ├── english_codemixed_to_hindi/
│   │   └── hindi_english_to_codemixed/
│   └── exp4_fewshot_learning/       # Exp5: Few-shot
│       ├── few5/
│       ├── few10/
│       ├── few20/
│       └── few50/
│
├── models/
│   ├── llama3.1_8b/
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── pretrain.py              # Pretraining script
│   │   ├── evaluate.py
│   │   ├── checkpoints/
│   │   │   ├── exp1_finetuning/     # Exp1 checkpoints
│   │   │   ├── exp2_pretraining/    # Exp2 checkpoints
│   │   │   └── exp3_full/           # Exp3 checkpoints
│   │   ├── results/
│   │   │   ├── exp1_results.json
│   │   │   ├── exp2_results.json
│   │   │   ├── exp3_results.json
│   │   │   ├── exp4_results.json
│   │   │   └── exp5_results.json
│   │   └── README.md
│   ├── mistral_7b/
│   │   └── [same structure]
│   ├── qwen2.5_7b/
│   │   └── [same structure]
│   ├── qwen2.5_1.5b/
│   │   └── [same structure]
│   ├── phi3_mini/
│   │   └── [same structure]
│   ├── xlmr_large/                  # Retrain for generation
│   │   └── [same structure]
│   └── muril_large/                 # Retrain for generation
│       └── [same structure]
│
├── evaluation/
│   ├── metrics/
│   │   ├── bleu.py
│   │   ├── rouge.py
│   │   ├── meteor.py
│   │   └── bertscore.py
│   ├── evaluate_all.py              # Evaluate all models
│   └── generate_tables.py          # Generate paper tables
│
└── results/
    ├── tables/
    │   ├── table1_overall_performance.md
    │   ├── table2_language_specific.md
    │   ├── table3_complexity_specific.md
    │   ├── table4_response_length.md
    │   └── table5_model_ranking.md
    └── paper_ready/
        └── complete_results.md
```

---

## 🚀 Implementation Plan (Revised)

### Phase 1: Dataset Preparation (Day 1-2)

**Task 1.1: Create 70/10/20 Split**
- [ ] Load all 1,200 samples
- [ ] Implement stratified split (language, complexity, bucket)
- [ ] Create train/val/test files
- [ ] Verify distribution preservation
- [ ] Save to `data/splits/`

**Task 1.2: Prepare Generation Format**
- [ ] Convert dialogue pairs to input/output format
- [ ] Format: `"User: {query}\nAssistant: {response}"`
- [ ] Create training scripts for each experiment

**Task 1.3: Prepare Pretraining Data**
- [ ] Collect legal corpus (POCSO Act, legal documents, case summaries)
- [ ] Preprocess and tokenize
- [ ] Save to `data/pretraining/legal_corpus/`

---

### Phase 2: Model Setup (Day 3-4)

**Task 2.1: Setup Generation Models**
- [ ] Create folder structure for each model
- [ ] Setup QLoRA for large models (LLaMA-3.1, Mistral, Qwen2.5-7B)
- [ ] Setup full fine-tuning for small models (Qwen2.5-1.5B, Phi-3)
- [ ] Create config files for each model
- [ ] Test model loading and inference

**Task 2.2: Setup Evaluation Framework**
- [ ] Implement BLEU calculator
- [ ] Implement ROUGE calculator
- [ ] Implement METEOR calculator
- [ ] Implement BERTScore calculator
- [ ] Create evaluation script

---

### Phase 3: Training - Experiment 1 (Day 5-8)

**Task 3.1: Exp1 - Finetuning Only (Baseline)**
- [ ] Train LLaMA-3.1-8B (QLoRA, direct finetuning)
- [ ] Train Mistral-7B (QLoRA, direct finetuning)
- [ ] Train Qwen2.5-7B (QLoRA, direct finetuning)
- [ ] Train Qwen2.5-1.5B (Full fine-tuning)
- [ ] Train Phi-3-mini (Full fine-tuning)
- [ ] Retrain XLM-RoBERTa-Large (for generation)
- [ ] Retrain MuRIL-Large (for generation)
- [ ] Evaluate all models on test set
- [ ] Save results to `models/{model}/results/exp1_results.json`

**Output**: Baseline performance for all models

---

### Phase 4: Training - Experiment 2 (Day 9-11)

**Task 4.1: Exp2 - Pretraining Only**
- [ ] Pretrain LLaMA-3.1-8B on legal corpus (CLM)
- [ ] Pretrain Mistral-7B on legal corpus (CLM)
- [ ] Pretrain Qwen2.5-7B on legal corpus (CLM)
- [ ] Pretrain Qwen2.5-1.5B on legal corpus (CLM)
- [ ] Pretrain Phi-3-mini on legal corpus (CLM)
- [ ] Save pretrained checkpoints

**Task 4.2: Exp2 - Zero-Shot Evaluation**
- [ ] Evaluate pretrained models on test set (no finetuning)
- [ ] Use instruction templates for generation
- [ ] Calculate metrics (BLEU, ROUGE, METEOR)
- [ ] Save results to `models/{model}/results/exp2_results.json`

**Output**: Pretraining-only performance

---

### Phase 5: Training - Experiment 3 (Day 12-15)

**Task 5.1: Exp3 - Pretraining + Finetuning (Full Pipeline)**
- [ ] Use pretrained checkpoints from Exp2
- [ ] Finetune LLaMA-3.1-8B on dialogue data
- [ ] Finetune Mistral-7B on dialogue data
- [ ] Finetune Qwen2.5-7B on dialogue data
- [ ] Finetune Qwen2.5-1.5B on dialogue data
- [ ] Finetune Phi-3-mini on dialogue data
- [ ] Evaluate all models on test set
- [ ] Save results to `models/{model}/results/exp3_results.json`

**Output**: Best performance (expected)

---

### Phase 6: Training - Experiment 4 (Day 16-17)

**Task 6.1: Exp4 - Zero-Shot Transfer**
- [ ] Create cross-lingual splits
- [ ] Train on Hindi+CodeMixed → Test on English
- [ ] Train on English+CodeMixed → Test on Hindi
- [ ] Train on Hindi+English → Test on CodeMixed
- [ ] Evaluate all models
- [ ] Save results to `models/{model}/results/exp4_results.json`

**Output**: Zero-shot transfer performance

---

### Phase 7: Training - Experiment 5 (Day 18)

**Task 7.1: Exp5 - Few-Shot Learning**
- [ ] Create few-shot splits (5, 10, 20, 50)
- [ ] Train all models with few-shot data
- [ ] Evaluate on full test set
- [ ] Save results to `models/{model}/results/exp5_results.json`

**Output**: Few-shot performance analysis

---

### Phase 8: Evaluation & Analysis (Day 19-20)

**Task 8.1: Comprehensive Evaluation**
- [ ] Evaluate all models on all experiments
- [ ] Calculate all metrics (BLEU, ROUGE, METEOR, BERTScore)
- [ ] Generate per-language analysis
- [ ] Generate per-complexity analysis
- [ ] Generate response length analysis
- [ ] Generate model ranking

**Task 8.2: Generate Paper Tables**
- [ ] Table 1: Overall Performance Metrics (Exp1, Exp2, Exp3 comparison)
- [ ] Table 2: Language-Specific Performance (ROUGE-1 F1 by language)
- [ ] Table 3: Complexity-Specific Performance (ROUGE-1 F1 by complexity)
- [ ] Table 4: Response Length Comparison
- [ ] Table 5: Model Ranking Summary (ROUGE-1 F1)

**Task 8.3: Ablation Study**
- [ ] Compare Exp1 (Finetuning) vs Exp2 (Pretraining) vs Exp3 (Both)
- [ ] Calculate improvement from pretraining
- [ ] Calculate improvement from finetuning
- [ ] Generate ablation study table

---

## 📊 Expected Results & Analysis

### Experiment Comparison

**Hypothesis:**
- **Exp1 (Finetuning only)**: Baseline performance
- **Exp2 (Pretraining only)**: Lower than Exp1 (no task-specific training)
- **Exp3 (Pretraining + Finetuning)**: **Best performance** (domain + task adaptation)

**Expected Ranking:**
1. **Exp3** (Pretraining + Finetuning) - Best
2. **Exp1** (Finetuning only) - Baseline
3. **Exp2** (Pretraining only) - Lower (no task finetuning)

### Ablation Study

**Research Question**: Does pretraining help?

**Comparison:**
- Exp3 vs Exp1: Improvement from pretraining
- Exp3 vs Exp2: Improvement from finetuning
- Exp1 vs Exp2: Finetuning vs Pretraining alone

**Expected Findings:**
- Pretraining + Finetuning > Finetuning only (Exp3 > Exp1)
- Pretraining + Finetuning > Pretraining only (Exp3 > Exp2)
- Finetuning only > Pretraining only (Exp1 > Exp2)

---

## 📝 Research Contributions

### Novel Contributions
1. **Ablation Study**: Pretraining vs Finetuning vs Both
2. **Multilingual Legal Dialogue Generation**: First comprehensive study on POCSO Act
3. **Zero-Shot Cross-Lingual Transfer**: Generation across Hindi, English, Code-mixed
4. **Complexity-Adaptive Generation**: Responses matching query complexity
5. **Comprehensive Evaluation**: BLEU, ROUGE, METEOR, BERTScore on legal domain

### Paper Structure
1. **Introduction**: Legal NLP, POCSO Act, multilingual challenges
2. **Related Work**: Legal NLP, dialogue systems, multilingual generation
3. **Dataset**: POCSO dialogues, 70/10/20 split, statistics
4. **Methodology**: Models, pretraining, finetuning, evaluation
5. **Experiments**: 
   - Exp1: Finetuning only (baseline)
   - Exp2: Pretraining only
   - Exp3: Pretraining + Finetuning (full pipeline)
   - Exp4: Zero-shot transfer
   - Exp5: Few-shot learning
6. **Results**: Tables, comparisons, ablation study
7. **Discussion**: Limitations, future work
8. **Conclusion**: Summary, contributions

---

## ✅ Success Criteria

1. ✅ All 7 models trained on Exp1, Exp2, Exp3
2. ✅ Ablation study completed (Exp1 vs Exp2 vs Exp3)
3. ✅ Zero-shot transfer evaluated (Exp4)
4. ✅ Few-shot learning evaluated (Exp5)
5. ✅ All metrics calculated (BLEU, ROUGE, METEOR, BERTScore)
6. ✅ All 5 tables generated (matching image format)
7. ✅ Pretraining + Finetuning > Finetuning only (Exp3 > Exp1)
8. ✅ Code repository clean and documented
9. ✅ Results ready for paper submission

---

## 📅 Timeline

**Week 1**: Dataset preparation, model setup  
**Week 2**: Training (Exp1, Exp2, Exp3)  
**Week 3**: Training (Exp4, Exp5), Evaluation  
**Week 4**: Analysis, documentation, paper writing

**Total**: 4 weeks for complete research pipeline

---

**Status**: 🟢 **READY TO START**  
**Next Step**: Create 70/10/20 dataset split

---

## 🎯 Key Differences from Original Plan

1. **Exp1**: Changed from "Supervised Baseline" to "Finetuning Only" (clearer)
2. **Exp2**: Changed from "Pretraining + Finetuning" to "Pretraining Only" (ablation)
3. **Exp3**: Changed to "Pretraining + Finetuning" (full pipeline)
4. **Exp4**: Zero-shot transfer (unchanged)
5. **Exp5**: Few-shot learning (unchanged)

**Result**: Clear ablation study structure (Exp1 vs Exp2 vs Exp3)

# 🎯 Comprehensive Research Plan: Top-Tier A* Paper
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
3. **Generation Models**: No seq2seq models trained (mT5/FLAN-T5 removed due to OOM)
4. **Pretraining**: No domain-specific pretraining
5. **Modern LLMs**: No evaluation of LLaMA-3.1, Mistral, Qwen2.5, Phi-3
6. **Generation Metrics**: No BLEU, ROUGE, METEOR scores

### Research Gap Identified

**Current**: Classification (understanding task)  
**Required**: Generation (response generation task)  
**Solution**: Dual-task approach (Understanding + Generation)

---

## 🎯 Research Objectives

### Primary Goals
1. **Task 1: Response Generation** - Generate legal responses to POCSO queries
2. **Task 2: Multilingual Support** - Hindi, English, Code-mixed generation
3. **Task 3: Complexity Adaptation** - Generate responses matching query complexity
4. **Task 4: Zero-Shot Transfer** - Cross-lingual generation without training
5. **Task 5: Few-Shot Learning** - Minimal data fine-tuning

### Secondary Goals
1. **Pretraining**: Domain-specific pretraining on legal corpus
2. **Finetuning**: Task-specific fine-tuning on dialogue pairs
3. **Evaluation**: Comprehensive metrics (BLEU, ROUGE, METEOR, BERTScore)

---

## 📁 Dataset Restructuring Plan

### Phase 1: Create 70/10/20 Train/Val/Test Split

**Current**: 80/20 train/test (1,200 samples)  
**New**: 70/10/20 train/val/test (1,200 samples)

**Stratified Split Strategy:**
```
Total: 1,200 samples
├── Train: 840 samples (70%)
│   ├── Hindi: 280 (23.33%)
│   ├── English: 280 (23.33%)
│   └── Code-mixed: 280 (23.33%)
├── Validation: 120 samples (10%)
│   ├── Hindi: 40 (3.33%)
│   ├── English: 40 (3.33%)
│   └── Code-mixed: 40 (3.33%)
└── Test: 240 samples (20%)
    ├── Hindi: 80 (6.67%)
    ├── English: 80 (6.67%)
    └── Code-mixed: 80 (6.67%)
```

**Stratification Levels:**
1. **Language** (Hindi/English/Code-mixed)
2. **Complexity** (Layman/Intermediate/Professional)
3. **Bucket** (A/B/C/D - turn count)
4. **Random seed**: 42 (reproducibility)

**What to Hide in Test Set:**
- ✅ **Hide**: Complete dialogue pairs (user query + assistant response)
- ✅ **Hide**: Complexity labels (for zero-shot evaluation)
- ✅ **Hide**: Language distribution (for cross-lingual evaluation)
- ✅ **Hide**: Turn count patterns (for generalization testing)

**Test Set Design (Research Best Practice):**
- **Temporal Split**: If data has timestamps, use chronological split
- **Stratified Split**: Maintain distribution across all dimensions
- **No Data Leakage**: Ensure no overlap between train/val/test
- **Blind Evaluation**: Test set only used for final evaluation

---

## 🤖 Model Selection & Architecture

### Tier 1: Generation Models (Primary Focus)

#### 1. **LLaMA-3.1-8B**
- **Rationale**: State-of-the-art multilingual LLM, strong Hindi support
- **Training**: QLoRA (4-bit quantization) for GPU efficiency
- **Use Case**: Primary generation model
- **Expected Performance**: Highest BLEU/ROUGE scores

#### 2. **Mistral-7B-Instruct-v0.3**
- **Rationale**: Strong instruction following, multilingual
- **Training**: QLoRA (4-bit quantization)
- **Use Case**: Instruction-tuned generation
- **Expected Performance**: High ROUGE-L (coherence)

#### 3. **Qwen2.5-7B**
- **Rationale**: Excellent multilingual performance, code-mixed handling
- **Training**: Full fine-tuning or QLoRA
- **Use Case**: Code-mixed text generation
- **Expected Performance**: Best code-mixed performance

#### 4. **Qwen2.5-1.5B**
- **Rationale**: Efficient baseline, fast inference
- **Training**: Full fine-tuning (fits in GPU)
- **Use Case**: Lightweight baseline
- **Expected Performance**: Lower but efficient

#### 5. **Phi-3-mini (3.8B)**
- **Rationale**: Small but capable, good for few-shot
- **Training**: Full fine-tuning
- **Use Case**: Few-shot learning experiments
- **Expected Performance**: Moderate, good for few-shot

### Tier 2: Encoder Models (Already Trained)

#### 6. **XLM-RoBERTa-Large** (Retrain with new split)
- **Status**: ✅ Already trained (classification)
- **Action**: Retrain with 70/10/20 split for generation task
- **Use Case**: Encoder baseline, retrieval component

#### 7. **MuRIL-Large** (Retrain with new split)
- **Status**: ✅ Already trained (classification)
- **Action**: Retrain with 70/10/20 split for generation task
- **Use Case**: Code-mixed understanding baseline

---

## 🏗️ Experimental Design

### Experiment 1: Supervised Baseline (70/10/20 Split)

**Purpose**: Establish upper bound with full training data

**Setup:**
- **Train**: 840 samples (70%)
- **Val**: 120 samples (10%)
- **Test**: 240 samples (20%)
- **Task**: Response generation (user query → assistant response)
- **Models**: All 7 models
- **Metrics**: BLEU-1/2/3/4, ROUGE-1/2/L F1, METEOR, BERTScore

**Expected Output:**
- Baseline performance for all models
- Best model identification
- Response quality assessment

---

### Experiment 2: Pretraining + Finetuning

**Purpose**: Domain adaptation through pretraining

**Phase 2.1: Pretraining (Exp2)**
- **Data**: Legal corpus (POCSO Act text, legal documents)
- **Task**: Masked Language Modeling (MLM) or Causal LM
- **Models**: LLaMA-3.1-8B, Mistral-7B, Qwen2.5-7B
- **Duration**: 1-2 epochs on legal corpus
- **Output**: Pretrained checkpoints

**Phase 2.2: Finetuning (Exp1)**
- **Data**: 70/10/20 split (dialogue pairs)
- **Task**: Response generation
- **Models**: Pretrained checkpoints from Phase 2.1
- **Output**: Final fine-tuned models

**Comparison:**
- **Exp1 (Finetuning only)** vs **Exp2 (Pretraining + Finetuning)**
- **Hypothesis**: Pretraining improves domain-specific generation

---

### Experiment 3: Zero-Shot Transfer

**Purpose**: Cross-lingual generation without training

**Scenarios:**
1. **Train on Hindi+CodeMixed → Test on English**
2. **Train on English+CodeMixed → Test on Hindi**
3. **Train on Hindi+English → Test on CodeMixed**

**Setup:**
- **Train**: 560 samples (2 languages × 280 each)
- **Val**: 80 samples (2 languages × 40 each)
- **Test**: 80 samples (target language)
- **Models**: All generation models
- **Metrics**: Language-specific ROUGE-1 F1, BLEU scores

**Expected Output:**
- Zero-shot transfer performance
- Best transfer direction identification
- Language-specific analysis

---

### Experiment 4: Few-Shot Learning

**Purpose**: Minimal data fine-tuning

**Shot Sizes**: 5, 10, 20, 50 examples per language

**Setup:**
- **Train**: N samples (few-shot size)
- **Val**: 20 samples
- **Test**: 240 samples (full test set)
- **Models**: All generation models
- **Metrics**: Performance vs shot size curve

**Expected Output:**
- Optimal few-shot size
- Model comparison for few-shot
- Efficiency analysis

---

## 📊 Evaluation Metrics (Matching Image Requirements)

### Primary Metrics (From Image)

1. **BLEU Scores**
   - BLEU-1 (unigram precision)
   - BLEU-2 (bigram precision)
   - BLEU-3 (trigram precision)
   - BLEU-4 (4-gram precision)

2. **ROUGE Scores**
   - ROUGE-1 F1 (unigram overlap)
   - ROUGE-2 F1 (bigram overlap)
   - ROUGE-L F1 (longest common subsequence)

3. **METEOR**
   - Semantic similarity metric

4. **Response Length Analysis**
   - Average reference length
   - Average candidate length
   - Ratio and difference

### Secondary Metrics

5. **BERTScore**
   - Semantic similarity using BERT embeddings

6. **Per-Language Performance**
   - ROUGE-1 F1 by language (English, Hindi, Code-mixed)

7. **Per-Complexity Performance**
   - ROUGE-1 F1 by complexity (Layman, Intermediate, Professional)

8. **Model Ranking**
   - Overall ranking by ROUGE-1 F1
   - Ranking by language
   - Ranking by complexity

---

## 🗂️ Repository Structure (High-Level Code)

```
legal-bot/
├── data/
│   ├── splits/
│   │   ├── train_70.jsonl          # 840 samples
│   │   ├── val_10.jsonl             # 120 samples
│   │   └── test_20.jsonl            # 240 samples
│   └── pretraining/
│       └── legal_corpus/            # Legal text for pretraining
│
├── experiments/
│   ├── exp1_supervised_baseline/    # 70/10/20 split
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── exp2_pretraining_finetuning/
│   │   ├── pretraining/             # Exp2: Pretraining phase
│   │   └── finetuning/              # Exp1: Finetuning phase
│   ├── exp3_zeroshot_transfer/
│   │   ├── hindi_codemixed_to_english/
│   │   ├── english_codemixed_to_hindi/
│   │   └── hindi_english_to_codemixed/
│   └── exp4_fewshot_learning/
│       ├── few5/
│       ├── few10/
│       ├── few20/
│       └── few50/
│
├── models/
│   ├── llama3.1_8b/
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── checkpoints/
│   │   ├── results/
│   │   │   ├── exp1_results.json
│   │   │   ├── exp2_results.json
│   │   │   ├── exp3_results.json
│   │   │   └── exp4_results.json
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
│   └── generate_tables.py           # Generate paper tables
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

## 🚀 Implementation Plan (Step-by-Step)

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
- [ ] Collect legal corpus (POCSO Act, legal documents)
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

### Phase 3: Training (Day 5-15)

**Task 3.1: Experiment 1 - Supervised Baseline**
- [ ] Train LLaMA-3.1-8B (QLoRA)
- [ ] Train Mistral-7B (QLoRA)
- [ ] Train Qwen2.5-7B (QLoRA)
- [ ] Train Qwen2.5-1.5B (Full fine-tuning)
- [ ] Train Phi-3-mini (Full fine-tuning)
- [ ] Retrain XLM-RoBERTa-Large (for generation)
- [ ] Retrain MuRIL-Large (for generation)
- [ ] Evaluate all models on test set
- [ ] Generate Table 1 (Overall Performance)

**Task 3.2: Experiment 2 - Pretraining + Finetuning**
- [ ] Pretrain LLaMA-3.1-8B on legal corpus (Exp2)
- [ ] Pretrain Mistral-7B on legal corpus (Exp2)
- [ ] Pretrain Qwen2.5-7B on legal corpus (Exp2)
- [ ] Finetune pretrained models on dialogue data (Exp1)
- [ ] Compare with Exp1 (finetuning only)
- [ ] Generate comparison results

**Task 3.3: Experiment 3 - Zero-Shot Transfer**
- [ ] Train on Hindi+CodeMixed → Test on English
- [ ] Train on English+CodeMixed → Test on Hindi
- [ ] Train on Hindi+English → Test on CodeMixed
- [ ] Evaluate all models
- [ ] Generate Table 2 (Language-Specific Performance)

**Task 3.4: Experiment 4 - Few-Shot Learning**
- [ ] Train with 5, 10, 20, 50 shots
- [ ] Evaluate all models
- [ ] Generate few-shot analysis

---

### Phase 4: Evaluation & Analysis (Day 16-18)

**Task 4.1: Comprehensive Evaluation**
- [ ] Evaluate all models on all experiments
- [ ] Calculate all metrics (BLEU, ROUGE, METEOR, BERTScore)
- [ ] Generate per-language analysis
- [ ] Generate per-complexity analysis
- [ ] Generate response length analysis
- [ ] Generate model ranking

**Task 4.2: Generate Paper Tables**
- [ ] Table 1: Overall Performance Metrics
- [ ] Table 2: Language-Specific Performance (ROUGE-1 F1)
- [ ] Table 3: Complexity-Specific Performance (ROUGE-1 F1)
- [ ] Table 4: Response Length Comparison
- [ ] Table 5: Model Ranking Summary (ROUGE-1 F1)

---

### Phase 5: Documentation (Day 19-20)

**Task 5.1: Results Documentation**
- [ ] Create comprehensive results document
- [ ] Document all experiments
- [ ] Create model comparison analysis
- [ ] Generate paper-ready tables

**Task 5.2: Code Documentation**
- [ ] Document all scripts
- [ ] Create README for each model
- [ ] Create usage examples
- [ ] Document evaluation metrics

---

## 🎯 Expected Outcomes

### Performance Targets (Based on Image)

**Best Model (LLaMA-3.1-8B):**
- BLEU-1: >0.30
- BLEU-4: >0.05
- ROUGE-1 F1: >0.35
- ROUGE-L F1: >0.23
- METEOR: >0.35

**Language-Specific (ROUGE-1 F1):**
- English: >0.40
- Hindi: >0.30
- Code-mixed: >0.30

**Complexity-Specific (ROUGE-1 F1):**
- Professional: >0.35
- Intermediate: >0.35
- Layman: >0.30

---

## 📝 Research Contributions

### Novel Contributions
1. **Multilingual Legal Dialogue Generation**: First comprehensive study on POCSO Act
2. **Zero-Shot Cross-Lingual Transfer**: Generation across Hindi, English, Code-mixed
3. **Complexity-Adaptive Generation**: Responses matching query complexity
4. **Pretraining + Finetuning**: Domain adaptation for legal NLP
5. **Comprehensive Evaluation**: BLEU, ROUGE, METEOR, BERTScore on legal domain

### Paper Structure
1. **Introduction**: Legal NLP, POCSO Act, multilingual challenges
2. **Related Work**: Legal NLP, dialogue systems, multilingual generation
3. **Dataset**: POCSO dialogues, 70/10/20 split, statistics
4. **Methodology**: Models, training, evaluation
5. **Experiments**: 4 experiments, results, analysis
6. **Results**: Tables, comparisons, insights
7. **Discussion**: Limitations, future work
8. **Conclusion**: Summary, contributions

---

## ⚠️ Critical Considerations

### Data Leakage Prevention
- ✅ **Strict Split**: No overlap between train/val/test
- ✅ **Temporal Split**: If possible, use chronological split
- ✅ **Blind Test**: Test set only for final evaluation
- ✅ **No Hyperparameter Tuning on Test**: Use validation set only

### Reproducibility
- ✅ **Random Seed**: 42 for all splits
- ✅ **Config Files**: All hyperparameters in YAML
- ✅ **Checkpoints**: Save all model checkpoints
- ✅ **Logs**: Save all training logs

### GPU Constraints
- ✅ **QLoRA**: Use 4-bit quantization for large models
- ✅ **Gradient Accumulation**: Simulate larger batch sizes
- ✅ **Mixed Precision**: FP16/BF16 training
- ✅ **Model Selection**: Prioritize models that fit GPU

---

## 🎓 Research Best Practices Applied

1. **Stratified Splitting**: Preserve distribution across all dimensions
2. **Multiple Baselines**: Compare against multiple models
3. **Comprehensive Metrics**: BLEU, ROUGE, METEOR, BERTScore
4. **Ablation Studies**: Pretraining vs finetuning only
5. **Zero-Shot Evaluation**: Test generalization
6. **Few-Shot Analysis**: Minimal data scenarios
7. **Per-Language Analysis**: Language-specific insights
8. **Per-Complexity Analysis**: Complexity-specific insights
9. **Response Length Analysis**: Generation quality assessment
10. **Model Ranking**: Clear performance comparison

---

## 📅 Timeline

**Week 1**: Dataset preparation, model setup  
**Week 2**: Training (Exp1, Exp2)  
**Week 3**: Training (Exp3, Exp4), Evaluation  
**Week 4**: Analysis, documentation, paper writing

**Total**: 4 weeks for complete research pipeline

---

## ✅ Success Criteria

1. ✅ All 7 models trained and evaluated
2. ✅ All 4 experiments completed
3. ✅ All metrics calculated (BLEU, ROUGE, METEOR, BERTScore)
4. ✅ All 5 tables generated (matching image format)
5. ✅ Zero-shot transfer >80% ROUGE-1 F1
6. ✅ Few-shot learning with 5-50 examples
7. ✅ Pretraining improves performance
8. ✅ Code repository clean and documented
9. ✅ Results ready for paper submission

---

**Status**: 🟢 **READY TO START**  
**Next Step**: Create 70/10/20 dataset split

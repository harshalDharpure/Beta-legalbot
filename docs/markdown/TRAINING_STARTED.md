# 🚀 Training Started - Status

## ✅ Training Initiated

**Date**: Current session  
**Experiment**: Exp1 (Finetuning Only - Baseline)  
**Status**: 🟢 **RUNNING**

---

## 📊 Models Training

### Currently Running (3 models on 3 GPUs)

| Model | GPU | Status | Log File |
|-------|-----|--------|----------|
| **qwen2.5_1.5b** | GPU 0 | 🟢 Running | `models/qwen2.5_1.5b/logs/training_gpu0.log` |
| **phi3_mini** | GPU 1 | 🟢 Running | `models/phi3_mini/logs/training_gpu1.log` |
| **qwen2.5_7b** | GPU 4 | 🟢 Running | `models/qwen2.5_7b/logs/training_gpu4.log` |

### Pending (4 models waiting for GPU availability)

| Model | Type | Notes |
|-------|------|-------|
| **mistral_7b** | QLoRA | Will start when GPU available |
| **llama3.1_8b** | QLoRA | Will start when GPU available |
| **xlmr_large** | Full fine-tuning | Will start when GPU available |
| **muril_large** | Full fine-tuning | Will start when GPU available |

---

## 🎮 GPU Status

**Free GPUs**: 0, 1, 4 (40GB each)  
**Busy GPUs**: 2, 3 (other processes)

---

## 📝 Monitor Training

### Check Status
```bash
python3 models/check_training_status.py
```

### View Logs
```bash
# Qwen2.5-1.5B
tail -f models/qwen2.5_1.5b/logs/training_gpu0.log

# Phi-3-mini
tail -f models/phi3_mini/logs/training_gpu1.log

# Qwen2.5-7B
tail -f models/qwen2.5_7b/logs/training_gpu4.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## 📊 Training Configuration

### Experiment: Exp1 (Finetuning Only)
- **Train Data**: 3,255 examples (70%)
- **Val Data**: 454 examples (10%)
- **Test Data**: 968 examples (20%)
- **Task**: Response Generation (user query → assistant response)

### Model Configurations

**Qwen2.5-1.5B**:
- Full fine-tuning (no QLoRA)
- Batch size: 8
- Gradient accumulation: 4
- Learning rate: 5e-5
- Epochs: 10

**Phi-3-mini**:
- Full fine-tuning (no QLoRA)
- Batch size: 8
- Gradient accumulation: 4
- Learning rate: 5e-5
- Epochs: 10

**Qwen2.5-7B**:
- QLoRA (4-bit quantization)
- Batch size: 2
- Gradient accumulation: 8
- Learning rate: 5e-5
- Epochs: 10

---

## ⏱️ Expected Training Time

- **Small models** (Qwen2.5-1.5B, Phi-3): ~2-4 hours
- **Medium models** (Qwen2.5-7B with QLoRA): ~4-6 hours
- **Large models** (LLaMA-3.1-8B, Mistral-7B with QLoRA): ~6-8 hours

---

## 🔄 Next Steps

1. **Monitor Training**: Check logs and GPU usage regularly
2. **Wait for Completion**: Small models will finish first
3. **Start Remaining Models**: When GPUs become available, start remaining models
4. **Evaluate**: After training, evaluate on test set
5. **Exp2 & Exp3**: Start pretraining and full pipeline experiments

---

## 📁 Files Created

- `models/train_generation_template.py` - Training script template
- `models/start_training_all_gpus.py` - GPU management script
- `models/check_training_status.py` - Status checking script
- `models/training_assignments.json` - GPU assignments

---

**Status**: 🟢 **TRAINING IN PROGRESS**  
**Last Updated**: Current session

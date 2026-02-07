# 📊 Current Training Status

**Last Updated**: Current Session

---

## ✅ **COMPLETED MODELS** (4/7 - 57%)

1. ✅ **Qwen2.5-7B** - COMPLETED
   - Checkpoint: `models/qwen2.5_7b/checkpoints/exp1/final`
   - Status: ✅ Fully trained

2. ✅ **Qwen2.5-1.5B** - COMPLETED
   - Checkpoint: `models/qwen2.5_1.5b/checkpoints/exp1/final`
   - Status: ✅ Fully trained

3. ✅ **XLM-RoBERTa-Large** - COMPLETED
   - Checkpoint: `models/xlmr_large/checkpoints/final`
   - Status: ✅ Fully trained

4. ✅ **MuRIL-Large** - COMPLETED
   - Checkpoint: `models/muril_large/checkpoints/final`
   - Status: ✅ Fully trained

---

## 🔄 **IN PROGRESS** (1/7 - 14%)

### 5. 🔄 **LLaMA-3.1-8B** - TRAINING ✅
- **Status**: ✅ **ACTIVELY TRAINING**
- **GPU**: 1 (Single GPU)
- **Progress**: **85/2040 steps (4.2%)**
- **Speed**: ~2.83 seconds/step
- **Estimated Time Remaining**: ~1.5 hours
- **GPU Memory**: 1,996 MB / 40 GB (68% utilization)
- **Log**: `models/llama3.1_8b/logs/training_single_gpu1.log`
- **Status**: ✅ Model loaded, QLoRA configured, training in progress

---

## ❌ **FAILED** (1/7 - 14%)

### 6. ❌ **Mistral-7B** - OOM ERROR
- **Status**: ❌ **OUT OF MEMORY ERROR**
- **GPUs**: 1, 4 (Multi-GPU setup)
- **Error**: `CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 39.49 GiB of which 11.88 MiB is free.`
- **Root Cause**: 
  - Mistral-7B with DataParallel is trying to use GPU 0 (which has LLaMA-3.1-8B running)
  - GPU 0 is already occupied by LLaMA (23.67 GiB used)
  - DataParallel conflict - needs different GPU assignment
- **Solution Needed**: 
  - Restart Mistral-7B on GPUs 3 and 4 (both are free)
  - Or wait for LLaMA to finish, then restart Mistral
- **Log**: `models/mistral_7b/logs/training_multi_gpu_1,4.log`

---

## ⏸️ **PENDING** (1/7 - 14%)

### 7. ⏸️ **Phi-3-mini** - PENDING
- **Status**: ⏸️ Not started (FP16 gradient issue needs fixing)
- **Action Needed**: Fix FP16 compatibility or disable FP16

---

## 🎮 **GPU STATUS**

| GPU | Memory Used | Memory Total | Utilization | Status | Assignment |
|-----|-------------|--------------|-------------|--------|------------|
| 0   | 963 MB      | 40 GB        | 24%         | 🟡 In Use | Other process |
| 1   | 1,996 MB    | 40 GB        | 68%         | 🟢 Training | **LLaMA-3.1-8B** |
| 2   | 36,372 MB   | 40 GB        | 100%        | 🔴 Full | Other process |
| 3   | 33,745 MB   | 40 GB        | 2%          | 🟡 In Use | Other process (idle) |
| 4   | 13 MB       | 40 GB        | 0%          | 🟢 Free | Available |

**Note**: 
- GPU 1 is actively training LLaMA-3.1-8B
- GPU 4 is free and available
- GPU 3 has memory allocated but low utilization (could be used for Mistral)

---

## 📈 **OVERALL PROGRESS**

- **Completed**: 4/7 models (57%)
- **In Progress**: 1/7 models (14%) - LLaMA training
- **Failed**: 1/7 models (14%) - Mistral OOM
- **Pending**: 1/7 models (14%) - Phi-3-mini

**Expected Timeline**:
- **LLaMA-3.1-8B**: ~1.5 hours remaining (85/2040 steps, 4.2% complete)
- **Mistral-7B**: Needs restart on different GPUs (3 and 4)
- **Phi-3-mini**: Needs FP16 fix

---

## 🔧 **IMMEDIATE ACTIONS NEEDED**

1. **Fix Mistral-7B OOM**:
   - Restart Mistral-7B on GPUs 3 and 4 (both available)
   - Or wait for LLaMA to finish, then use GPU 1 and 4

2. **Monitor LLaMA Training**:
   - Currently at step 85/2040 (4.2%)
   - Training speed: ~2.83s/step
   - Estimated completion: ~1.5 hours

3. **Fix Phi-3-mini**:
   - Address FP16 gradient issue
   - Can start after other models complete

---

## 📝 **MONITORING COMMANDS**

### Check LLaMA Progress:
```bash
tail -f models/llama3.1_8b/logs/training_single_gpu1.log
```

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

### Check Process Status:
```bash
ps aux | grep train_generation_template.py | grep -v grep
```

### Check Training Steps:
```bash
grep -E "Step|loss" models/llama3.1_8b/logs/training_single_gpu1.log | tail -10
```

---

## ⚠️ **ISSUES & RESOLUTIONS**

### Issue 1: Mistral-7B OOM
- **Problem**: DataParallel trying to use GPU 0 (occupied by LLaMA)
- **Solution**: Restart on GPUs 3 and 4
- **Command**:
  ```bash
  cd /DATA/vaneet_2221cs15/legal-bot && \
  CUDA_VISIBLE_DEVICES=3,4 python3 models/train_generation_template.py \
    --model mistral_7b --experiment exp1 --gpu 3 4 --multi-gpu \
    2>&1 | tee models/mistral_7b/logs/training_multi_gpu_3,4.log &
  ```

---

**Status**: 🟡 **1 MODEL TRAINING, 1 MODEL NEEDS RESTART**

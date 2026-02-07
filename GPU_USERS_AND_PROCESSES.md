# 🖥️ GPU Users and Processes Report

## 📊 Summary

**Total GPU Users**: 4 users
**Active GPU Processes**: 2 processes (non-training)
**Zombie Processes**: 0 ✅
**Training Processes**: 0 (Qwen2.5-7B may have stopped)

---

## 👥 GPU Users List

### 1. **shahbaz+** (Primary GPU User)
- **GPU 2**: PID 3690491 - Python process (33.4GB memory)
- **GPU 3**: PID 3688952 - Python process `con_Net_aut_02.py` (33.7GB memory)
- **Total Python Processes**: 7 processes
- **Status**: Active (using 2 GPUs heavily)

### 2. **saba_24+**
- **Total Python Processes**: 3 processes
- **GPU Usage**: Not directly using GPUs (may be waiting)

### 3. **ajit_24+**
- **Total Python Processes**: 1 process
- **GPU Usage**: Not directly using GPUs

### 4. **root** (System)
- **Processes**: System processes (Xorg, nvidia drivers)
- **GPU Usage**: Xorg using 4MB on each GPU (normal)

### 5. **vaneet_2221cs15** (You)
- **Training Processes**: 0 (Qwen2.5-7B may have stopped)
- **Status**: No active training

---

## 🎮 GPU Usage Details

| GPU | User | PID | Process | Memory | Utilization |
|-----|------|-----|---------|--------|-------------|
| **GPU 0** | root | 3477 | Xorg | 4MB | 0% |
| **GPU 1** | root | 3477 | Xorg | 4MB | 0% |
| **GPU 2** | root | 3477 | Xorg | 4MB | 0% |
| **GPU 2** | **shahbaz+** | 3690491 | python | **33.4GB** | **48%** |
| **GPU 3** | root | 3477 | Xorg | 4MB | 0% |
| **GPU 3** | **shahbaz+** | 3688952 | python (con_Net_aut_02.py) | **33.7GB** | **0%** |
| **GPU 4** | root | 3477 | Xorg | 4MB | 0% |

---

## ⚠️ Issues Found

### 1. **Qwen2.5-7B Training Stopped**
- **Expected**: Should be running on GPU 0, 1
- **Actual**: No training process found
- **Action**: Check logs and restart if needed

### 2. **Heavy GPU Usage by shahbaz+**
- **GPU 2**: 33.4GB / 40GB (83% memory, 48% utilization)
- **GPU 3**: 33.7GB / 40GB (84% memory, 0% utilization - may be idle)
- **Note**: This is blocking your training on GPUs 2, 3

### 3. **GPU 3 Process May Be Idle**
- PID 3688952: 0% utilization but 33.7GB memory
- **Possible**: Process holding memory but not actively computing
- **Action**: Check if process is actually needed

---

## ✅ Zombie Processes

**Status**: ✅ **No zombie processes found**

All processes are healthy.

---

## 🔍 Unnecessary Processes

### Potentially Unnecessary:
1. **PID 3688952** (GPU 3): 
   - 0% GPU utilization but holding 33.7GB
   - Process: `python -i con_Net_aut_02.py`
   - **May be idle/interactive session**

### System Processes (Normal):
- Xorg (4MB per GPU) - Normal display server
- nvidia driver processes - Normal system processes

---

## 📋 Recommendations

1. **Check Qwen2.5-7B Training**:
   ```bash
   tail -50 models/qwen2.5_7b/logs/training_multi_gpu_0,1.log
   ```

2. **Contact shahbaz+** if GPU 3 process is idle:
   - Process holding 33.7GB but 0% utilization
   - May be blocking your training

3. **Use Available GPUs**:
   - GPU 0: Free (10MB used)
   - GPU 1: Free (10MB used)
   - GPU 4: Free (10MB used)
   - **Can start training on these GPUs**

---

## 📊 Process Summary

| User | Python Processes | GPU Usage | Status |
|------|------------------|-----------|--------|
| **shahbaz+** | 7 | GPU 2, 3 (heavy) | Active |
| **saba_24+** | 3 | None | Waiting |
| **ajit_24+** | 1 | None | Waiting |
| **vaneet_2221cs15** | 0 | None | No training |

---

**Last Updated**: Current session  
**Status**: 4 users, 0 zombies, 2 heavy GPU processes (shahbaz+)

# Training Status

## ✅ Training Started Successfully!

### Models Running:
1. **mT5-Large** - GPU 0 ✅ RUNNING
2. **XLM-RoBERTa-Large** - GPU 1 ✅ RUNNING (restarted)
3. **MuRIL-Large** - GPU 2 ✅ RUNNING
4. **FLAN-T5-XL** - GPU 3 ✅ RUNNING

### GPU Status:
- 5x NVIDIA A100-PCIE-40GB GPUs available
- Models distributed across GPUs 0-3
- GPU 4 available for additional training if needed

### Monitoring Commands:

```bash
# Check training status
python3 models/check_status.py

# Monitor specific model log
tail -f models/mt5_large/logs/training_gpu0.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check all processes
ps aux | grep train.py
```

### Process Information:
Saved in: `models/training_processes.json`

### Notes:
- All models are training in background
- Logs are saved in each model's `logs/` directory
- Training will continue until completion or manual stop
- Use `pkill -f train.py` to stop all training

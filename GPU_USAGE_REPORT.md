# 🖥️ GPU Usage Report

## 📊 GPU Users and Processes

This report shows who is using GPUs and identifies any zombie or unnecessary processes.

---

## 🔍 Analysis Commands

Run these commands to check GPU usage:
```bash
# Check GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory,user --format=csv

# Check all processes
nvidia-smi

# Check for zombie processes
ps aux | awk '$8 ~ /^Z/'

# Check training processes
ps aux | grep train_generation_template.py
```

---

**Last Updated**: Current session

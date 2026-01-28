"""
Start training all models in background, distributing across available GPUs
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def get_gpu_info():
    """Get GPU information"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.free,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_free': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'utilization': int(parts[4])
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def find_free_gpus(gpus, min_memory_gb=10):
    """Find GPUs with low utilization and sufficient memory"""
    free_gpus = []
    for gpu in gpus:
        memory_free_gb = gpu['memory_free'] / 1024
        if gpu['utilization'] < 20 and memory_free_gb >= min_memory_gb:
            free_gpus.append(gpu)
    return free_gpus

def start_training_on_gpu(model_key, gpu_id, script_path):
    """Start training on specific GPU"""
    # Get absolute paths
    script_abs = os.path.abspath(script_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_abs)))  # Go to project root
    
    log_file = os.path.join(base_dir, f"models/{model_key}/logs/training_gpu{gpu_id}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create training command with GPU assignment
    cmd = [
        sys.executable, script_abs
    ]
    
    # Set CUDA device
    env = os.environ.copy()
    if gpu_id is not None and gpu_id >= 0:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Start process in background
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=base_dir
        )
    
    return process, log_file

def main():
    print("="*70)
    print("STARTING TRAINING ON AVAILABLE GPUs")
    print("="*70)
    
    # Get GPU information
    print("\n1. Checking GPU availability...")
    gpus = get_gpu_info()
    
    if not gpus:
        print("❌ No GPUs found or nvidia-smi not available")
        print("   Starting training on CPU (will be slow)")
        gpu_assignments = {0: None}  # Use CPU
    else:
        print(f"   Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            memory_gb = gpu['memory_free'] / 1024
            print(f"   GPU {gpu['index']}: {gpu['name']}, "
                  f"Free: {memory_gb:.1f}GB/{gpu['memory_total']/1024:.1f}GB, "
                  f"Utilization: {gpu['utilization']}%")
        
        # Find free GPUs
        free_gpus = find_free_gpus(gpus)
        print(f"\n   Free GPUs: {len(free_gpus)}")
        
        if not free_gpus:
            print("   ⚠️  No free GPUs found, but proceeding anyway...")
            free_gpus = gpus[:4]  # Use first 4 GPUs anyway
        
        gpu_assignments = {i: gpu['index'] for i, gpu in enumerate(free_gpus[:4])}
    
    # Models to train
    models = {
        'mt5_large': {
            'name': 'mT5-Large',
            'script': 'models/mt5_large/train.py',
            'gpu_memory': 24
        },
        'xlmr_large': {
            'name': 'XLM-RoBERTa-Large',
            'script': 'models/xlmr_large/train.py',
            'gpu_memory': 12
        },
        'muril_large': {
            'name': 'MuRIL-Large',
            'script': 'models/muril_large/train.py',
            'gpu_memory': 12
        },
        'flan_t5_xl': {
            'name': 'FLAN-T5-XL',
            'script': 'models/flan_t5_xl/train.py',
            'gpu_memory': 24
        }
    }
    
    # Assign models to GPUs
    print("\n2. Assigning models to GPUs...")
    processes = []
    gpu_idx = 0
    
    for model_key, model_info in models.items():
        if gpu_idx < len(gpu_assignments):
            assigned_gpu = gpu_assignments[gpu_idx]
            if assigned_gpu is not None:
                print(f"   {model_info['name']:25} → GPU {assigned_gpu}")
            else:
                print(f"   {model_info['name']:25} → CPU")
            
            script_path = model_info['script']
            if os.path.exists(script_path):
                process, log_file = start_training_on_gpu(
                    model_key, 
                    assigned_gpu if assigned_gpu is not None else -1,
                    script_path
                )
                processes.append({
                    'model': model_key,
                    'name': model_info['name'],
                    'process': process,
                    'gpu': assigned_gpu,
                    'log_file': log_file,
                    'pid': process.pid
                })
                print(f"      Started (PID: {process.pid}, Log: {log_file})")
            else:
                print(f"      ❌ Script not found: {script_path}")
            
            gpu_idx += 1
        else:
            print(f"   {model_info['name']:25} → Waiting for GPU availability")
    
    # Save process information
    processes_info = {
        'started_at': datetime.now().isoformat(),
        'processes': [
            {
                'model': p['model'],
                'name': p['name'],
                'pid': p['pid'],
                'gpu': p['gpu'],
                'log_file': p['log_file']
            }
            for p in processes
        ]
    }
    
    info_file = 'models/training_processes.json'
    with open(info_file, 'w') as f:
        json.dump(processes_info, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    print(f"\nStarted {len(processes)} training processes:")
    for p in processes:
        gpu_info = f"GPU {p['gpu']}" if p['gpu'] is not None else "CPU"
        print(f"  {p['name']:25} - PID: {p['pid']:6} - {gpu_info:8} - {p['log_file']}")
    
    print(f"\nProcess information saved to: {info_file}")
    print("\nTo monitor training:")
    print("  tail -f models/{model_name}/logs/training_gpu*.log")
    print("\nTo check GPU usage:")
    print("  watch -n 1 nvidia-smi")
    print("\nTo stop training:")
    print("  pkill -f train.py")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()

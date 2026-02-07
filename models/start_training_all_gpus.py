"""
Start training on all free GPUs.
Checks GPU availability and starts training on free GPUs.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path


def get_gpu_status():
    """Get GPU status."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_free = int(parts[1])
                utilization = int(parts[2])
                gpus.append({
                    'id': gpu_id,
                    'memory_free_mb': memory_free,
                    'utilization': utilization
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU status: {e}")
        return []


def find_free_gpus(min_memory_gb=20, max_utilization=10):
    """Find free GPUs."""
    gpus = get_gpu_status()
    free_gpus = []
    
    for gpu in gpus:
        memory_gb = gpu['memory_free_mb'] / 1024
        if memory_gb >= min_memory_gb and gpu['utilization'] <= max_utilization:
            free_gpus.append(gpu['id'])
    
    return free_gpus


def start_training_on_gpu(model_name, gpu_id, experiment='exp1'):
    """Start training on a specific GPU."""
    script_path = os.path.join('models', 'train_generation_template.py')
    
    cmd = [
        sys.executable,
        script_path,
        '--model', model_name,
        '--experiment', experiment,
        '--gpu', str(gpu_id)
    ]
    
    log_file = os.path.join('models', model_name, 'logs', f'training_gpu{gpu_id}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    print(f"🚀 Starting {model_name} on GPU {gpu_id}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Log: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
        )
    
    return process, log_file


def main():
    """Main function to start training on free GPUs."""
    print("="*70)
    print("Starting Training on Free GPUs")
    print("="*70)
    
    # Models to train (prioritize small models first)
    models = [
        'qwen2.5_1.5b',  # Small, fast
        'phi3_mini',     # Small, fast
        'qwen2.5_7b',    # Medium (QLoRA)
        'mistral_7b',    # Medium (QLoRA)
        'llama3.1_8b',   # Large (QLoRA)
        'xlmr_large',    # Encoder (retrain)
        'muril_large'    # Encoder (retrain)
    ]
    
    # Find free GPUs
    free_gpus = find_free_gpus(min_memory_gb=20, max_utilization=10)
    
    if not free_gpus:
        print("❌ No free GPUs found!")
        print("   Requirements: >=20GB free memory, <=10% utilization")
        return
    
    print(f"\n✅ Found {len(free_gpus)} free GPU(s): {free_gpus}")
    
    # Start training on free GPUs
    processes = []
    gpu_assignments = {}
    
    for i, model_name in enumerate(models):
        if i < len(free_gpus):
            gpu_id = free_gpus[i]
            process, log_file = start_training_on_gpu(model_name, gpu_id, experiment='exp1')
            processes.append((model_name, process, log_file, gpu_id))
            gpu_assignments[model_name] = gpu_id
            time.sleep(5)  # Stagger starts
    
    # Save assignments
    assignments_file = 'models/training_assignments.json'
    with open(assignments_file, 'w') as f:
        json.dump(gpu_assignments, f, indent=2)
    
    print(f"\n✅ Started {len(processes)} training processes")
    print(f"\n📊 GPU Assignments:")
    for model_name, process, log_file, gpu_id in processes:
        print(f"   {model_name:20} → GPU {gpu_id} (log: {log_file})")
    
    print(f"\n💡 Monitor training:")
    print(f"   tail -f {log_file}")
    print(f"\n💡 Check GPU usage:")
    print(f"   watch -n 1 nvidia-smi")
    
    print(f"\n📝 Assignments saved to: {assignments_file}")


if __name__ == "__main__":
    main()

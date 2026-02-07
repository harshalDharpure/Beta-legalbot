"""
Start training with multi-GPU support.
Automatically detects free GPUs and uses multiple GPUs per model for faster training.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path


def find_free_gpus(min_memory_gb=30, max_utilization=10):
    """Find free GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        free_gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_free_mb = int(parts[1])
                utilization = int(parts[2])
                memory_free_gb = memory_free_mb / 1024
                
                if memory_free_gb >= min_memory_gb and utilization <= max_utilization:
                    free_gpus.append(gpu_id)
        
        return free_gpus
    except Exception as e:
        print(f"Error finding free GPUs: {e}")
        return []


def start_multi_gpu_training(model_name, gpu_ids, experiment='exp1'):
    """Start training on multiple GPUs using DataParallel."""
    script_path = os.path.join('models', 'train_generation_template.py')
    gpu_str = ','.join(map(str, gpu_ids))
    num_gpus = len(gpu_ids)
    
    log_file = os.path.join('models', model_name, 'logs', f'training_multi_gpu_{gpu_str}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Simple command - Trainer handles multi-GPU automatically
    cmd = [
        sys.executable,
        script_path,
        '--model', model_name,
        '--experiment', experiment,
        '--gpu'] + [str(g) for g in gpu_ids] + ['--multi-gpu'
    ]
    
    print(f"🚀 Starting {model_name} on GPUs {gpu_ids} ({num_gpus} GPUs)")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Log: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_str}
        )
    
    return process, log_file


def is_model_trained(model_name, experiment='exp1'):
    """Check if model training is already complete."""
    final_checkpoint = f"models/{model_name}/checkpoints/{experiment}/final"
    return os.path.exists(final_checkpoint) and os.path.exists(f"{final_checkpoint}/model.safetensors")


def start_single_model_multi_gpu(model_name, experiment='exp1', max_gpus_per_model=4):
    """Start a single model training on multiple GPUs."""
    # Check if already trained
    if is_model_trained(model_name, experiment):
        print(f"⏭️  {model_name} already trained - skipping")
        return None, None, None
    
    free_gpus = find_free_gpus()
    
    if not free_gpus:
        print(f"❌ No free GPUs for {model_name}")
        return None, None, None
    
    # Use up to max_gpus_per_model GPUs
    gpu_ids = free_gpus[:max_gpus_per_model]
    
    process, log_file = start_multi_gpu_training(model_name, gpu_ids, experiment)
    return process, log_file, gpu_ids


def main():
    """Main function to start multi-GPU training."""
    print("="*70)
    print("Starting Multi-GPU Training")
    print("="*70)
    
    # Models to train (prioritize QLoRA models - more memory efficient)
    models = [
        ('qwen2.5_1.5b', 2),      # Small model - 2 GPUs ✅ COMPLETE
        ('qwen2.5_7b', 2),         # Medium model - 2 GPUs (with QLoRA - memory efficient)
        ('mistral_7b', 2),         # Medium model - 2 GPUs (with QLoRA - memory efficient)
        ('llama3.1_8b', 3),        # Large model - 3 GPUs (with QLoRA - memory efficient)
        ('phi3_mini', 1),          # Small model - 1 GPU (skip for now - OOM issues)
    ]
    
    free_gpus = find_free_gpus()
    print(f"\n✅ Found {len(free_gpus)} free GPU(s): {free_gpus}")
    
    if len(free_gpus) < 2:
        print("⚠️ Need at least 2 free GPUs for multi-GPU training")
        print("   Falling back to single-GPU training...")
        # Use single GPU training script instead
        os.system(f"python3 models/start_training_all_gpus.py")
        return
    
    processes = []
    gpu_assignments = {}
    used_gpus = set()
    
    for model_name, num_gpus_needed in models:
        # Check if already trained
        if is_model_trained(model_name, experiment='exp1'):
            print(f"⏭️  {model_name} already trained - skipping")
            continue
        
        # Find available GPUs not yet used
        available_gpus = [g for g in free_gpus if g not in used_gpus]
        
        if len(available_gpus) < num_gpus_needed:
            print(f"⚠️ Not enough GPUs for {model_name} (needs {num_gpus_needed}, have {len(available_gpus)})")
            continue
        
        # Assign GPUs
        gpu_ids = available_gpus[:num_gpus_needed]
        used_gpus.update(gpu_ids)
        
        process, log_file, assigned_gpus = start_single_model_multi_gpu(
            model_name, 
            experiment='exp1',
            max_gpus_per_model=num_gpus_needed
        )
        
        if process and assigned_gpus:
            processes.append((model_name, process, log_file, assigned_gpus))
            gpu_assignments[model_name] = assigned_gpus
            time.sleep(5)  # Stagger starts
    
    # Save assignments
    assignments_file = 'models/multi_gpu_assignments.json'
    with open(assignments_file, 'w') as f:
        json.dump(gpu_assignments, f, indent=2)
    
    print(f"\n✅ Started {len(processes)} training processes with multi-GPU")
    print(f"\n📊 GPU Assignments:")
    for model_name, process, log_file, gpu_ids in processes:
        print(f"   {model_name:20} → GPUs {gpu_ids} ({len(gpu_ids)} GPUs)")
        print(f"                      Log: {log_file}")
    
    print(f"\n💡 Monitor training:")
    print(f"   tail -f {log_file}")
    print(f"\n💡 Check GPU usage:")
    print(f"   watch -n 1 nvidia-smi")
    
    print(f"\n📝 Assignments saved to: {assignments_file}")


if __name__ == "__main__":
    main()

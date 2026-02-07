"""
Check training status for all running models.
"""

import os
import json
import subprocess
from pathlib import Path


def check_gpu_usage():
    """Check current GPU usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except:
        return "Error getting GPU status"


def check_training_processes():
    """Check running training processes."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        
        processes = []
        for line in result.stdout.split('\n'):
            if 'train_generation_template.py' in line and 'grep' not in line:
                processes.append(line)
        return processes
    except:
        return []


def check_log_files():
    """Check log files for recent activity."""
    models_dir = Path('models')
    log_files = []
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            logs_dir = model_dir / 'logs'
            if logs_dir.exists():
                for log_file in logs_dir.glob('training_gpu*.log'):
                    if log_file.exists():
                        size = log_file.stat().st_size
                        log_files.append({
                            'model': model_dir.name,
                            'log': str(log_file),
                            'size': size
                        })
    
    return log_files


def main():
    """Main function to check training status."""
    print("="*70)
    print("Training Status Check")
    print("="*70)
    
    # Check GPU assignments
    assignments_file = Path('models/training_assignments.json')
    if assignments_file.exists():
        with open(assignments_file, 'r') as f:
            assignments = json.load(f)
        
        print("\n📊 GPU Assignments:")
        for model, gpu_id in assignments.items():
            print(f"   {model:20} → GPU {gpu_id}")
    else:
        print("\n⚠️ No assignments file found")
    
    # Check running processes
    print("\n🔄 Running Processes:")
    processes = check_training_processes()
    if processes:
        for proc in processes:
            parts = proc.split()
            pid = parts[1]
            cmd = ' '.join(parts[10:])
            print(f"   PID {pid}: {cmd[:60]}...")
    else:
        print("   No training processes found")
    
    # Check log files
    print("\n📝 Log Files:")
    log_files = check_log_files()
    if log_files:
        for log_info in log_files:
            size_kb = log_info['size'] / 1024
            print(f"   {log_info['model']:20} → {log_info['log']} ({size_kb:.1f} KB)")
    else:
        print("   No log files found")
    
    # Check GPU usage
    print("\n🎮 GPU Usage:")
    gpu_info = check_gpu_usage()
    if gpu_info:
        print(gpu_info)
    else:
        print("   Error getting GPU info")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

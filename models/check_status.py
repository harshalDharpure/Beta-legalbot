"""
Check training status for all models
"""

import os
import json
import subprocess
from datetime import datetime

def check_process_status(pid):
    """Check if process is still running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_log_tail(log_file, lines=5):
    """Get last few lines of log file"""
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return ''.join(all_lines[-lines:])
    except:
        return "Log file not found or empty"

def main():
    info_file = 'models/training_processes.json'
    
    if not os.path.exists(info_file):
        print("No training processes found. Start training first.")
        return
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    print("="*70)
    print("TRAINING STATUS")
    print("="*70)
    print(f"\nStarted at: {info['started_at']}")
    print(f"Current time: {datetime.now().isoformat()}")
    
    print("\n" + "-"*70)
    print("Process Status:")
    print("-"*70)
    
    all_running = True
    for proc in info['processes']:
        is_running = check_process_status(proc['pid'])
        status = "✅ RUNNING" if is_running else "❌ STOPPED"
        gpu_info = f"GPU {proc['gpu']}" if proc['gpu'] is not None else "CPU"
        
        print(f"\n{proc['name']:25}")
        print(f"  Status: {status}")
        print(f"  PID: {proc['pid']}")
        print(f"  GPU: {gpu_info}")
        print(f"  Log: {proc['log_file']}")
        
        if is_running:
            log_tail = get_log_tail(proc['log_file'], 3)
            if log_tail.strip():
                print(f"  Last log lines:")
                for line in log_tail.strip().split('\n')[-3:]:
                    if line.strip():
                        print(f"    {line[:70]}")
        
        if not is_running:
            all_running = False
    
    print("\n" + "-"*70)
    print("GPU Usage:")
    print("-"*70)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except:
        print("Could not get GPU info")
    
    print("="*70)
    
    if all_running:
        print("✅ All training processes are running!")
    else:
        print("⚠️  Some training processes have stopped")

if __name__ == '__main__':
    main()

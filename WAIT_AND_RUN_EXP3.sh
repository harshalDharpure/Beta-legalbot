#!/bin/bash
# Wait until a GPU has enough free memory, then run Exp3 training.
# Run in background: nohup ./WAIT_AND_RUN_EXP3.sh > wait_exp3.log 2>&1 &
#
# Consider "free": memory.used < FREE_MEM_MB (default 8000 = 8GB free on 40GB GPU)
# Poll every POLL_SEC seconds (default 120).

set -e
cd "$(dirname "$0")"
FREE_MEM_MB=${FREE_MEM_MB:-8000}   # Need at least this much free (40 - used < 8 => used < 32)
POLL_SEC=${POLL_SEC:-120}
LOG="models/exp3_training_logs/wait_and_run_exp3.log"
mkdir -p models/exp3_training_logs

echo "[$(date -Iseconds)] Waiting for a GPU with used < ${FREE_MEM_MB} MiB (~$(( 40 - FREE_MEM_MB / 1024 )) GB free). Poll every ${POLL_SEC}s. Log: $LOG"

while true; do
  # nvidia-smi: index, memory.used (MiB), utilization.gpu (%)
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    gpu_id=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
    mem_used=$(echo "$line" | cut -d',' -f3 | tr -d ' ')
    util=$(echo "$line" | cut -d',' -f5 | tr -d ' ')
    if [[ -n "$mem_used" && "$mem_used" -lt "$FREE_MEM_MB" ]] 2>/dev/null; then
      echo "[$(date -Iseconds)] GPU $gpu_id is free enough (used ${mem_used} MiB, util ${util}%). Starting Exp3 training."
      export CUDA_VISIBLE_DEVICES=$gpu_id
      exec bash START_EXP3_TRAINING.sh 2>&1 | tee -a "$LOG"
      exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  echo "[$(date -Iseconds)] No free GPU yet. Sleeping ${POLL_SEC}s..."
  sleep "$POLL_SEC"
done

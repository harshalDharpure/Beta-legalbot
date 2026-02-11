#!/bin/bash
# Run Exp3 evaluation for all 5 models in parallel on multiple GPUs.
# Usage: GPUS="0 1 4" ./START_EXP3_EVALUATION.sh

set -e
cd "$(dirname "$0")"
GPUS="${GPUS:-0 1 2 3 4}"
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
LOG_DIR="models/exp3_evaluation_logs"
mkdir -p "$LOG_DIR"
gpu_array=($GPUS)
num_gpus=${#gpu_array[@]}
num_models=${#MODELS[@]}

echo "=============================================="
echo "Exp3 Evaluation (pretrain + finetune)"
echo "GPUs: $GPUS ($num_gpus GPU(s))"
echo "Models: ${MODELS[*]}"
echo "=============================================="

run_one() {
    local model=$1
    local gpu_id=$2
    local logfile="$LOG_DIR/exp3_${model}.log"
    echo "[GPU $gpu_id] Starting $model -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 models/evaluate_generation.py --model "$model" --experiment exp3 \
        2>&1 | tee "$logfile" || true
    echo "[GPU $gpu_id] Finished $model"
}

if [ "$num_gpus" -ge "$num_models" ]; then
    pids=()
    for i in "${!MODELS[@]}"; do
        run_one "${MODELS[i]}" "${gpu_array[i]}" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
else
    i=0
    while [ $i -lt $num_models ]; do
        pids=()
        for j in $(seq 0 $((num_gpus - 1))); do
            [ $((i + j)) -lt $num_models ] || break
            run_one "${MODELS[i+j]}" "${gpu_array[j]}" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        i=$((i + num_gpus))
    done
fi
echo ""
echo "Exp3 evaluation complete. Results: models/*/results/exp3_results.json"
echo "Logs: $LOG_DIR/"

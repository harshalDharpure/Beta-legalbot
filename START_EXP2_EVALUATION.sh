#!/bin/bash
# Run Exp2 evaluation for all 5 models in parallel on multiple GPUs.
# Each model runs on one GPU. Set GPUS to a space-separated list of free GPU IDs.
#
# Usage:
#   ./START_EXP2_EVALUATION.sh              # use GPUs 0,1,2,3,4
#   GPUS="1 2 4" ./START_EXP2_EVALUATION.sh  # use only GPUs 1, 2, 4
#   GPUS="0" ./START_EXP2_EVALUATION.sh      # sequential on one GPU

set -e
cd "$(dirname "$0")"

# Space-separated list of GPU IDs to use (default: 0 1 2 3 4)
GPUS="${GPUS:-0 1 2 3 4}"
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
LOG_DIR="models/exp2_evaluation_logs"
mkdir -p "$LOG_DIR"

# Convert GPUS string to array
gpu_array=($GPUS)
num_gpus=${#gpu_array[@]}
num_models=${#MODELS[@]}

echo "=============================================="
echo "Exp2 Evaluation (zero-shot on test set)"
echo "GPUs: $GPUS ($num_gpus GPU(s))"
echo "Models: ${MODELS[*]}"
echo "=============================================="

run_one() {
    local model=$1
    local gpu_id=$2
    local logfile="$LOG_DIR/exp2_${model}.log"
    echo "[GPU $gpu_id] Starting $model -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu_id python3 models/evaluate_generation.py --model "$model" --experiment exp2 \
        2>&1 | tee "$logfile" || true
    echo "[GPU $gpu_id] Finished $model"
}

if [ "$num_gpus" -ge "$num_models" ]; then
    # One model per GPU, all in parallel
    pids=()
    for i in "${!MODELS[@]}"; do
        gpu_id=${gpu_array[i]}
        run_one "${MODELS[i]}" "$gpu_id" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
else
    # Fewer GPUs than models: run in waves (num_gpus models at a time in parallel)
    i=0
    while [ $i -lt $num_models ]; do
        pids=()
        for j in $(seq 0 $((num_gpus - 1))); do
            [ $((i + j)) -lt $num_models ] || break
            model=${MODELS[i+j]}
            gpu_id=${gpu_array[j]}
            run_one "$model" "$gpu_id" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        i=$((i + num_gpus))
    done
fi

echo ""
echo "=============================================="
echo "Exp2 evaluation complete. Results:"
echo "  models/*/results/exp2_results.json"
echo "Logs: $LOG_DIR/"
echo "=============================================="

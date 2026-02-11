#!/bin/bash
# Run all Exp4 + Exp5 jobs distributed across multiple GPUs.
# Usage: GPUS="0 1 4" nohup ./RUN_ALL_EXP4_AND_EXP5_MULTI_GPU.sh > models/exp4_exp5_logs/multi_gpu.log 2>&1 &
# Or: GPUS="0 1 2 3 4" ./RUN_ALL_EXP4_AND_EXP5_MULTI_GPU.sh
set -e
cd "$(dirname "$0")"
GPUS="${GPUS:-0 1 4}"
LOG_DIR="models/exp4_exp5_logs"
JOBS_DIR="$LOG_DIR/jobs"
mkdir -p "$LOG_DIR" "$JOBS_DIR"

# Build full job list: one line per job. Format: "exp4 MODEL CONFIG" or "exp5 MODEL FEW DIR"
JOBLIST="$JOBS_DIR/all_jobs.txt"
> "$JOBLIST"
MODELS=(qwen2.5_1.5b phi3_mini qwen2.5_7b mistral_7b llama3.1_8b)
EXP4_CONFIGS=(hindi_code_mixed_to_english english_code_mixed_to_hindi hindi_english_to_code_mixed)
FEW_SIZES=(5 10 20 50)
EXP5_DIRECTIONS=(hindi_code_mixed_to_english english_code_mixed_to_hindi)

for model in "${MODELS[@]}"; do
  for config in "${EXP4_CONFIGS[@]}"; do
    echo "exp4 $model $config" >> "$JOBLIST"
  done
  for few in "${FEW_SIZES[@]}"; do
    for dir in "${EXP5_DIRECTIONS[@]}"; do
      echo "exp5 $model $few $dir" >> "$JOBLIST"
    done
  done
done
total_jobs=$(wc -l < "$JOBLIST")
echo "Total jobs: $total_jobs"

# Split jobs round-robin across GPUs
gpu_array=($GPUS)
num_gpus=${#gpu_array[@]}
for ((i=0; i<num_gpus; i++)); do
  > "$JOBS_DIR/gpu_${i}.txt"
done
idx=0
while IFS= read -r line; do
  g=$((idx % num_gpus))
  echo "$line" >> "$JOBS_DIR/gpu_${g}.txt"
  ((idx++)) || true
done < "$JOBLIST"
echo "Split into $num_gpus GPU queues: $(wc -l "$JOBS_DIR"/gpu_*.txt | head -n -1)"

run_one_job() {
  local gpu_id=$1
  local line=$2
  local typ model rest
  read -r typ model rest <<< "$line"
  export CUDA_VISIBLE_DEVICES=$gpu_id
  if [ "$typ" = "exp4" ]; then
    local config=$rest
    local out="models/$model/results/exp4_${config}_results.json"
    if [ -f "$out" ]; then echo "[GPU$gpu_id] SKIP exp4 $model $config"; return 0; fi
    echo "[GPU$gpu_id] $(date) Exp4 $model $config"
    python3 models/train_generation_template.py --model "$model" --experiment exp4 --config "$config" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp4 --config "$config" 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
  else
    local few dir
    few="${rest%% *}"
    dir="${rest#* }"
    local out="models/$model/results/exp5_few${few}_${dir}_results.json"
    if [ -f "$out" ]; then echo "[GPU$gpu_id] SKIP exp5 $model few$few $dir"; return 0; fi
    echo "[GPU$gpu_id] $(date) Exp5 $model few$few $dir"
    python3 models/train_generation_template.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
  fi
  echo "[GPU$gpu_id] Done: $line"
}

worker() {
  local idx=$1
  local gpu_id=$2
  local jobfile="$JOBS_DIR/gpu_${idx}.txt"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    run_one_job "$gpu_id" "$line"
  done < "$jobfile"
  echo "[GPU$gpu_id] Worker finished $(date)"
}

echo "========== MULTI-GPU RUN started $(date) GPUS=$GPUS =========="
for ((i=0; i<num_gpus; i++)); do
  gpu_id=${gpu_array[i]}
  worker "$i" "$gpu_id" &
  echo "Started worker for GPU $gpu_id (queue gpu_${i}.txt) PID $!"
done
wait
echo "========== MULTI-GPU RUN finished $(date) =========="

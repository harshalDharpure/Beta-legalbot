#!/bin/bash
# Run Exp4 (3 configs) and Exp5 (4 few × 2 directions) for NEW models only: qwen3_8b, gemma3_4b, gemma3_12b.
# Exp4: 3 × 3 = 9 jobs. Exp5: 3 × 4 × 2 = 24 jobs. Total 33 jobs.
# Usage:
#   GPU=0 ./RUN_EXP4_EXP5_NEW_MODELS.sh              # single GPU, sequential
#   GPUS="0 1 2" nohup ./RUN_EXP4_EXP5_NEW_MODELS.sh > models/exp4_exp5_logs/new_models.log 2>&1 &
set -e
cd "$(dirname "$0")"
GPUS="${GPUS:-0}"
LOG_DIR="models/exp4_exp5_logs"
JOBS_DIR="$LOG_DIR/jobs"
mkdir -p "$LOG_DIR" "$JOBS_DIR"

NEW_MODELS=(qwen3_8b gemma3_4b gemma3_12b)
EXP4_CONFIGS=(hindi_code_mixed_to_english english_code_mixed_to_hindi hindi_english_to_code_mixed)
FEW_SIZES=(5 10 20 50)
EXP5_DIRECTIONS=(hindi_code_mixed_to_english english_code_mixed_to_hindi)

# Optional: fewer epochs for faster run (default use config epochs)
export EXP4_EXP5_EPOCHS="${EXP4_EXP5_EPOCHS:-}"

JOBLIST="$JOBS_DIR/new_models_exp4_exp5.txt"
> "$JOBLIST"

for model in "${NEW_MODELS[@]}"; do
  for config in "${EXP4_CONFIGS[@]}"; do
    out="models/$model/results/exp4_${config}_results.json"
    if [ -f "$out" ]; then echo "[SKIP] Exp4 $model $config (exists)"; continue; fi
    echo "exp4 $model $config" >> "$JOBLIST"
  done
  for few in "${FEW_SIZES[@]}"; do
    for dir in "${EXP5_DIRECTIONS[@]}"; do
      out="models/$model/results/exp5_few${few}_${dir}_results.json"
      if [ -f "$out" ]; then echo "[SKIP] Exp5 $model few$few $dir (exists)"; continue; fi
      echo "exp5 $model $few $dir" >> "$JOBLIST"
    done
  done
done

n=$(wc -l < "$JOBLIST" 2>/dev/null || echo 0)
if [ "$n" -eq 0 ]; then
  echo "All Exp4/Exp5 jobs for new models already done. Run update script to refresh table:"
  echo "  python3 models/evaluation_results/update_all_results_in_table.py"
  exit 0
fi

echo "New models Exp4+Exp5: $n jobs to run. GPUS=$GPUS"

gpu_array=($GPUS)
num_gpus=${#gpu_array[@]}
for ((i=0; i<num_gpus; i++)); do > "$JOBS_DIR/gpu_new_${i}.txt"; done
idx=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  g=$((idx % num_gpus))
  echo "$line" >> "$JOBS_DIR/gpu_new_${g}.txt"
  ((idx++)) || true
done < "$JOBLIST"

run_one_job() {
  local gpu_id=$1
  local line=$2
  local typ model rest
  read -r typ model rest <<< "$line"
  export CUDA_VISIBLE_DEVICES=$gpu_id
  if [ "$typ" = "exp4" ]; then
    local config=$rest
    echo "[GPU$gpu_id] $(date) Exp4 $model $config"
    python3 models/train_generation_template.py --model "$model" --experiment exp4 --config "$config" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp4 --config "$config" 2>&1 | tee -a "$LOG_DIR/exp4_${model}_${config}.log" || true
  else
    local few dir
    few="${rest%% *}"
    dir="${rest#* }"
    echo "[GPU$gpu_id] $(date) Exp5 $model few$few $dir"
    python3 models/train_generation_template.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" --gpu 0 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
    python3 models/evaluate_generation.py --model "$model" --experiment exp5 --few-size "$few" --direction "$dir" 2>&1 | tee -a "$LOG_DIR/exp5_${model}_few${few}_${dir}.log" || true
  fi
  echo "[GPU$gpu_id] Done: $line"
}

worker() {
  local idx=$1
  local gpu_id=$2
  local jobfile="$JOBS_DIR/gpu_new_${idx}.txt"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    run_one_job "$gpu_id" "$line"
  done < "$jobfile"
  echo "[GPU$gpu_id] Worker finished $(date)"
}

echo "========== NEW MODELS Exp4+Exp5 started $(date) =========="
for ((i=0; i<num_gpus; i++)); do
  gpu_id=${gpu_array[i]}
  worker "$i" "$gpu_id" &
done
wait
echo "========== NEW MODELS Exp4+Exp5 finished $(date) =========="
echo "Updating category_wise_results_all_experiments.md..."
python3 models/evaluation_results/update_all_results_in_table.py
echo "Done. Table updated."

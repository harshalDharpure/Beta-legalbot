#!/bin/bash
# Monitor training progress

echo "============================================================"
echo "TRAINING MONITOR"
echo "============================================================"

echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "Running Training Processes:"
ps aux | grep train.py | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15}'

echo ""
echo "Recent Log Activity:"
for log in models/*/logs/training_gpu*.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "--- $(basename $(dirname $(dirname $log))) ---"
        tail -3 "$log" 2>/dev/null || echo "No content yet"
    fi
done

echo ""
echo "============================================================"

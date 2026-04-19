# Qwen 3 8B Instruct

## Model Information

- **Model Name**: Qwen/Qwen3-8B
- **Type**: Generation Model (latest Qwen3 family; different from Qwen2.5 7B/1.5B already in experiments)
- **Quantization**: 4bit
- **QLoRA**: True
- **GPU**: 1× 40GB recommended

## Training

From repo root:

### Exp1: Finetuning Only
```bash
python models/train_generation_template.py --model qwen3_8b --experiment exp1 --gpu 0
```

### Other experiments (Exp3, Exp4, Exp5)
```bash
python models/train_generation_template.py --model qwen3_8b --experiment exp3 --gpu 0
```

## Evaluation
```bash
python models/evaluate_generation.py --model qwen3_8b --experiment exp1
```

## Results

Results are saved in `results/` (created at runtime).

# Gemma 3 4B Instruct

## Model Information

- **Model Name**: google/gemma-3-4b-it
- **Type**: Generation Model (latest Gemma 3 family)
- **Quantization**: 4bit
- **QLoRA**: True
- **GPU**: 1× 40GB recommended

## Training

From repo root:

### Exp1: Finetuning Only
```bash
python models/train_generation_template.py --model gemma3_4b --experiment exp1 --gpu 0
```

### Other experiments (Exp3, Exp4, Exp5)
```bash
python models/train_generation_template.py --model gemma3_4b --experiment exp3 --gpu 0
```

## Evaluation
```bash
python models/evaluate_generation.py --model gemma3_4b --experiment exp1
```

## Results

Results are saved in `results/` (created at runtime).

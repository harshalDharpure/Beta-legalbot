# Llama 4 Scout 17B-16E Instruct

## Model Information

- **Model Name**: meta-llama/Llama-4-Scout-17B-16E-Instruct
- **Type**: Generation Model (latest Llama 4 family; MoE, 17B active params; different from LLaMA 3.1 8B already in experiments)
- **Quantization**: 4bit
- **QLoRA**: True
- **GPU**: 1× 40GB recommended

## Access

Requires Meta Llama 4 license and Hugging Face access approval. Accept the license at the model page and log in with `huggingface-cli login` before training.

## Training

From repo root:

### Exp1: Finetuning Only
```bash
python models/train_generation_template.py --model llama4_scout --experiment exp1 --gpu 0
```

### Other experiments (Exp3, Exp4, Exp5)
```bash
python models/train_generation_template.py --model llama4_scout --experiment exp3 --gpu 0
```

## Evaluation
```bash
python models/evaluate_generation.py --model llama4_scout --experiment exp1
```

## Results

Results are saved in `results/` (created at runtime).

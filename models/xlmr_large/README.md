# XLM-RoBERTa-Large

## Model Information

- **Model Name**: xlm-roberta-large
- **Type**: Encoder (for generation task)
- **Quantization**: none
- **QLoRA**: False

## Training

### Exp1: Finetuning Only
```bash
python train.py --experiment exp1
```

### Exp2: Pretraining Only
```bash
python pretrain.py --experiment exp2
python evaluate.py --experiment exp2 --checkpoint checkpoints/exp2_pretraining/
```

### Exp3: Pretraining + Finetuning
```bash
python pretrain.py --experiment exp3
python train.py --experiment exp3 --checkpoint checkpoints/exp3_pretraining/
```

## Results

Results are saved in `results/` directory:
- `exp1_results.json` - Exp1 results
- `exp2_results.json` - Exp2 results
- `exp3_results.json` - Exp3 results

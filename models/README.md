# Model Training Infrastructure

Professional training setup for POCSO Legal Dialogue research with separate folders for each model.

## Structure

```
models/
├── README.md                    # This file
├── config.yaml                  # Common configuration
├── requirements.txt             # Dependencies
├── mt5_large/                   # mT5-Large model
│   ├── config.yaml
│   ├── train.py
│   ├── checkpoints/
│   └── logs/
├── xlmr_large/                  # XLM-RoBERTa-Large
│   ├── config.yaml
│   ├── train.py
│   ├── checkpoints/
│   └── logs/
├── muril_large/                 # MuRIL-Large (Code-mixed)
│   ├── config.yaml
│   ├── train.py
│   ├── checkpoints/
│   └── logs/
└── flan_t5_xl/                  # FLAN-T5-XL (Zero-shot/Few-shot)
    ├── config.yaml
    ├── train.py
    ├── checkpoints/
    └── logs/
```

## Models

### 1. mT5-Large (`mt5_large/`)
- **Type**: Seq2Seq (Generation)
- **Use**: Primary model for all experiments
- **GPU**: ~24GB
- **Training**: `python models/mt5_large/train.py`

### 2. XLM-RoBERTa-Large (`xlmr_large/`)
- **Type**: Encoder (Understanding)
- **Use**: Baseline encoder, multilingual understanding
- **GPU**: ~12GB
- **Training**: `python models/xlmr_large/train.py`

### 3. MuRIL-Large (`muril_large/`)
- **Type**: Encoder (Code-mixed specific)
- **Use**: Code-mixed experiments, Hindi-English mixing
- **GPU**: ~12GB
- **Training**: `python models/muril_large/train.py`

### 4. FLAN-T5-XL (`flan_t5_xl/`)
- **Type**: Seq2Seq (Zero-shot/Few-shot)
- **Use**: Zero-shot and few-shot experiments
- **GPU**: ~24GB
- **Training**: `python models/flan_t5_xl/train.py`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r models/requirements.txt
```

### 2. Train a Model
```bash
# Train mT5-Large
cd models/mt5_large
python train.py

# Train XLM-R-Large
cd models/xlmr_large
python train.py

# Train MuRIL-Large
cd models/muril_large
python train.py

# Train FLAN-T5-XL
cd models/flan_t5_xl
python train.py
```

### 3. Monitor Training
```bash
# TensorBoard logs
tensorboard --logdir models/{model_name}/logs
```

## Configuration

Each model has its own `config.yaml` with:
- Model-specific parameters
- Training hyperparameters
- Data paths
- Output directories

Common settings are in `models/config.yaml` and can be inherited.

## Training Scripts

Each model folder contains:
- `train.py`: Main training script
- `config.yaml`: Model-specific configuration
- `checkpoints/`: Saved model checkpoints
- `logs/`: Training logs (TensorBoard)

## Notes

- All scripts use the same experimental data structure
- Configs are model-specific but follow same structure
- Checkpoints are saved per model in separate directories
- Logs are organized per model for easy comparison

# Code Mixed POCSO Dataset

This dataset contains code-mixed (Hindi-English) language dialogues related to POCSO (Protection of Children from Sexual Offences) Act cases, organized by complexity level and turn count buckets.

## Dataset Structure

```
code_mixed_posco_dataset/
├── layman/
│   ├── bucket_A.jsonl      (34 entries, 2-3 turns)
│   ├── bucket_B.jsonl      (33 entries, 3-4 turns)
│   ├── bucket_C.jsonl      (33 entries, 4-5 turns)
│   └── bucket_D.jsonl      (33 entries, 5-6 turns)
├── intermediate/
│   ├── bucket_A.jsonl      (33 entries, 2-3 turns)
│   ├── bucket_B.jsonl      (33 entries, 3-4 turns)
│   ├── bucket_C.jsonl      (33 entries, 4-5 turns)
│   └── bucket_D.jsonl      (34 entries, 5-6 turns)
└── professional/
    ├── bucket_A.jsonl      (33 entries, 2-3 turns)
    ├── bucket_B.jsonl      (34 entries, 3-4 turns)
    ├── bucket_C.jsonl      (34 entries, 4-5 turns)
    └── bucket_D.jsonl      (33 entries, 5-6 turns)
```

## Distribution Summary

| Complexity   | Bucket A (2-3) | Bucket B (3-4) | Bucket C (4-5) | Bucket D (5-6) | Subtotal |
|--------------|----------------|----------------|----------------|----------------|----------|
| Layman       | 34             | 33             | 33             | 33             | 133      |
| Intermediate | 33             | 33             | 33             | 34             | 133      |
| Professional | 33             | 34             | 34             | 33             | 134      |
| **TOTAL**    | **100**        | **100**        | **100**        | **100**        | **400**  |

## Data Format

Each entry in the JSONL files follows this structure:

```json
{
  "dialogue_id": "CM_A_C0801_001",
  "language": "code_mixed",
  "complexity": "layman|intermediate|professional",
  "turn_count": 3,
  "turns": [
    {
      "role": "user|assistant",
      "text": "..."
    }
  ],
  "statutes_cited": ["POCSO Act Section 3", "IPC Section 376", ...],
  "bucket": "A|B|C|D",
  "case_id": 801
}
```

## Bucket Definitions

- **Bucket A**: 2-3 turns per dialogue
- **Bucket B**: 3-4 turns per dialogue
- **Bucket C**: 4-5 turns per dialogue
- **Bucket D**: 5-6 turns per dialogue

## Complexity Levels

- **Layman**: Simple language, basic legal concepts explained in everyday terms (code-mixed)
- **Intermediate**: Moderate complexity, some legal terminology used (code-mixed)
- **Professional**: Advanced legal language and concepts (code-mixed)

## Language Characteristics

This dataset contains code-mixed Hindi-English text, where speakers naturally mix Hindi and English words/phrases in their conversations, which is common in Indian multilingual contexts.

## Usage

### Reading a single file:
```python
import json

with open('code_mixed_posco_dataset/layman/bucket_A.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        print(entry['dialogue_id'])
```

### Using the helper script:
```python
from load_dataset import load_all_data, load_by_complexity, load_data

# Load all data
all_data = load_all_data()

# Load by complexity
layman_data = load_by_complexity('layman')

# Load specific combination
data = load_data(complexity='layman', bucket='A')
```

## Statistics

- **Total Entries**: 400
- **Language**: Code-mixed (Hindi-English)
- **Domain**: Legal (POCSO Act)
- **Format**: JSONL (JSON Lines)

## all_cases.txt (Legal corpus for Exp2 pretraining)

The file **all_cases.txt** contains raw legal judgment text (court cases, IPC, POCSO, etc.) with `[case N]` section markers. It is **suitable for Exp2/Exp3 pretraining** as a legal domain corpus (continued pretraining on raw text). To use it:

```bash
python data/prepare_legal_corpus.py --use-all-cases
```

This copies `all_cases.txt` into `experiments/exp2_pretraining_only/pretraining/legal_corpus/` and the Exp3 corpus directory. Exp1 fine-tuning uses the dialogue splits (train_70/val_10/test_20.jsonl), not this file.

## Notes

- All dialogues are in code-mixed Hindi-English
- Each entry represents a complete legal consultation dialogue
- Statutes cited include relevant sections from POCSO Act, IPC, CrPC, and other relevant laws
- Bucket distribution ensures balanced representation across turn counts

# English POCSO Dataset

This dataset contains English language dialogues related to POCSO (Protection of Children from Sexual Offences) Act cases, organized by complexity level and turn count buckets.

## Dataset Structure

```
english_posco_dataset/
├── layman/
│   ├── bucket_A.jsonl      (33 entries, 2-3 turns)
│   ├── bucket_B.jsonl      (34 entries, 3-4 turns)
│   ├── bucket_C.jsonl      (33 entries, 4-5 turns)
│   └── bucket_D.jsonl      (33 entries, 5-6 turns)
├── intermediate/
│   ├── bucket_A.jsonl      (33 entries, 2-3 turns)
│   ├── bucket_B.jsonl      (33 entries, 3-4 turns)
│   ├── bucket_C.jsonl      (34 entries, 4-5 turns)
│   └── bucket_D.jsonl      (33 entries, 5-6 turns)
└── professional/
    ├── bucket_A.jsonl      (34 entries, 2-3 turns)
    ├── bucket_B.jsonl      (33 entries, 3-4 turns)
    ├── bucket_C.jsonl      (33 entries, 4-5 turns)
    └── bucket_D.jsonl      (34 entries, 5-6 turns)
```

## Distribution Summary

| Complexity   | Bucket A (2-3) | Bucket B (3-4) | Bucket C (4-5) | Bucket D (5-6) | Subtotal |
|--------------|----------------|----------------|----------------|----------------|----------|
| Layman       | 33             | 34             | 33             | 33             | 133      |
| Intermediate | 33             | 33             | 34             | 33             | 133      |
| Professional | 34             | 33             | 33             | 34             | 134      |
| **TOTAL**    | **100**        | **100**        | **100**        | **100**        | **400**  |

## Data Format

Each entry in the JSONL files follows this structure:

```json
{
  "dialogue_id": "EN_A_C0401_001",
  "language": "english",
  "complexity": "layman|intermediate|professional",
  "turn_count": 2,
  "turns": [
    {
      "role": "user|assistant",
      "text": "..."
    }
  ],
  "statutes_cited": ["POCSO Act Section 3", "IPC Section 376", ...],
  "bucket": "A|B|C|D",
  "case_id": 401
}
```

## Bucket Definitions

- **Bucket A**: 2-3 turns per dialogue
- **Bucket B**: 3-4 turns per dialogue
- **Bucket C**: 4-5 turns per dialogue
- **Bucket D**: 5-6 turns per dialogue

## Complexity Levels

- **Layman**: Simple language, basic legal concepts explained in everyday terms
- **Intermediate**: Moderate complexity, some legal terminology used
- **Professional**: Advanced legal language and concepts

## Usage

### Reading a single file:
```python
import json

with open('english_posco_dataset/layman/bucket_A.jsonl', 'r', encoding='utf-8') as f:
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
- **Language**: English
- **Domain**: Legal (POCSO Act)
- **Format**: JSONL (JSON Lines)

## Notes

- All dialogues are in English
- Each entry represents a complete legal consultation dialogue
- Statutes cited include relevant sections from POCSO Act, IPC, CrPC, and other relevant laws
- Bucket distribution ensures balanced representation across turn counts

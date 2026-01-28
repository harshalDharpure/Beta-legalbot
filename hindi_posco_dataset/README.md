# Hindi POCSO Dataset

This dataset contains Hindi language dialogues related to POCSO (Protection of Children from Sexual Offences) Act cases, organized by complexity level and turn count buckets.

## Dataset Structure

```
hindi_posco_dataset/
├── layman/
│   ├── bucket_A.jsonl      (33 entries, 2-3 turns)
│   ├── bucket_B.jsonl      (33 entries, 3-4 turns)
│   ├── bucket_C.jsonl      (34 entries, 4-5 turns)
│   └── bucket_D.jsonl      (33 entries, 5-6 turns)
├── intermediate/
│   ├── bucket_A.jsonl      (34 entries, 2-3 turns)
│   ├── bucket_B.jsonl      (33 entries, 3-4 turns)
│   ├── bucket_C.jsonl      (33 entries, 4-5 turns)
│   └── bucket_D.jsonl      (33 entries, 5-6 turns)
└── professional/
    ├── bucket_A.jsonl      (33 entries, 2-3 turns)
    ├── bucket_B.jsonl      (34 entries, 3-4 turns)
    ├── bucket_C.jsonl      (34 entries, 4-5 turns)
    └── bucket_D.jsonl      (33 entries, 5-6 turns)
```

## Distribution Summary

| Complexity   | Bucket A (2-3) | Bucket B (3-4) | Bucket C (4-5) | Bucket D (5-6) | Subtotal |
|--------------|----------------|----------------|----------------|----------------|----------|
| Layman       | 33             | 33             | 34             | 33             | 133      |
| Intermediate | 34             | 33             | 33             | 33             | 133      |
| Professional | 33             | 34             | 34             | 33             | 134      |
| **TOTAL**    | **100**        | **100**        | **101**        | **99**         | **400**  |

## Data Format

Each entry in the JSONL files follows this structure:

```json
{
  "dialogue_id": "HN_A_C0001_001",
  "language": "hindi",
  "complexity": "layman|intermediate|professional",
  "turn_count": 3,
  "turns": [
    {
      "role": "user|assistant",
      "text": "..."
    }
  ],
  "statutes_cited": ["POCSO धारा 3-5, 19, 24", "IPC धारा 376", ...],
  "bucket": "A|B|C|D",
  "case_id": 1
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

with open('hindi_posco_dataset/layman/bucket_A.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        print(entry['dialogue_id'])
```

### Reading all files in a complexity level:
```python
import json
import os

complexity = 'layman'
data = []

for bucket in ['A', 'B', 'C', 'D']:
    filepath = f'hindi_posco_dataset/{complexity}/bucket_{bucket}.jsonl'
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
```

## Statistics

- **Total Entries**: 400
- **Language**: Hindi
- **Domain**: Legal (POCSO Act)
- **Format**: JSONL (JSON Lines)

## Notes

- All dialogues are in Hindi (Devanagari script)
- Each entry represents a complete legal consultation dialogue
- Statutes cited include relevant sections from POCSO Act, IPC, CrPC, and other relevant laws
- Bucket distribution ensures balanced representation across turn counts

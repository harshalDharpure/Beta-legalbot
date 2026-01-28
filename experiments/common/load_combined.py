"""Load combined dataset (all languages)"""

import json
import os

def load_combined_dataset():
    """Load the complete combined dataset"""
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, 'combined_dataset.jsonl')
    
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

if __name__ == '__main__':
    data = load_combined_dataset()
    print(f"Combined dataset: {len(data)} entries")

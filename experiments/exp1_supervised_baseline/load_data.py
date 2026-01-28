"""Load data for Experiment 1: Supervised Baseline"""

import json
import os

def load_jsonl(filepath):
    """Load data from JSONL file"""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def load_train_test():
    """Load train and test splits"""
    base_dir = os.path.dirname(__file__)
    train_data = load_jsonl(os.path.join(base_dir, 'data', 'train.jsonl'))
    test_data = load_jsonl(os.path.join(base_dir, 'data', 'test.jsonl'))
    return train_data, test_data

def load_combined():
    """Load complete combined dataset"""
    base_dir = os.path.dirname(__file__)
    return load_jsonl(os.path.join(base_dir, 'data', 'combined.jsonl'))

if __name__ == '__main__':
    train, test = load_train_test()
    print(f"Train: {len(train)} entries")
    print(f"Test: {len(test)} entries")

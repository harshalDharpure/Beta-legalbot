"""Load data for Experiment 2: Monolingual Baselines"""

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

def load_language_data(language):
    """Load train and test data for a specific language"""
    base_dir = os.path.dirname(__file__)
    train_data = load_jsonl(os.path.join(base_dir, 'data', f'{language}_train.jsonl'))
    test_data = load_jsonl(os.path.join(base_dir, 'data', f'{language}_test.jsonl'))
    return train_data, test_data

def get_available_languages():
    """Get list of available languages"""
    return ['hindi', 'code_mixed', 'english']

if __name__ == '__main__':
    for lang in get_available_languages():
        train, test = load_language_data(lang)
        print(f"{lang:12}: Train {len(train):3}, Test {len(test):3}")

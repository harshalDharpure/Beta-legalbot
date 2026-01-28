"""Load data for Experiment 3: Zero-Shot Cross-Lingual Transfer"""

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

def load_zeroshot_config(config_name):
    """Load train and test data for a zero-shot configuration"""
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, 'data', config_name, 'train.jsonl')
    test_path = os.path.join(base_dir, 'data', config_name, 'test.jsonl')
    
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    
    return train_data, test_data

def get_available_configs():
    """Get list of available zero-shot configurations"""
    return [
        'hindi_code_mixed_to_english',
        'english_code_mixed_to_hindi',
        'hindi_english_to_code_mixed'
    ]

if __name__ == '__main__':
    print("Available Zero-Shot Configurations:")
    for config in get_available_configs():
        train, test = load_zeroshot_config(config)
        print(f"  {config:35}: Train {len(train):3}, Test {len(test):3}")

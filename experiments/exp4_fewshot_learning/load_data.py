"""Load data for Experiment 4: Few-Shot Cross-Lingual Learning"""

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

def load_fewshot_config(few_size, direction):
    """
    Load train and test data for a few-shot configuration.
    
    Args:
        few_size: Number of few-shot examples (5, 10, 20, or 50)
        direction: 'hindi_code_mixed_to_english' or 'english_code_mixed_to_hindi'
    """
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, 'data', f'few{few_size}', direction, 'train.jsonl')
    test_path = os.path.join(base_dir, 'data', f'few{few_size}', direction, 'test.jsonl')
    
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    
    return train_data, test_data

def get_available_fewshot_sizes():
    """Get list of available few-shot sizes"""
    return [5, 10, 20, 50]

def get_available_directions():
    """Get list of available directions"""
    return [
        'hindi_code_mixed_to_english',
        'english_code_mixed_to_hindi'
    ]

if __name__ == '__main__':
    print("Available Few-Shot Configurations:")
    for size in get_available_fewshot_sizes():
        for direction in get_available_directions():
            train, test = load_fewshot_config(size, direction)
            print(f"  Few{size:2} {direction:35}: Train {len(train):3}, Test {len(test):3}")

"""
Create 70/10/20 train/val/test split for generation task.

Stratified by:
- Language (Hindi, English, Code-mixed)
- Complexity (Layman, Intermediate, Professional)
- Bucket (A, B, C, D)

Output format: Generation task (user query → assistant response)
"""

import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# Set random seed for reproducibility
random.seed(42)


def load_all_datasets() -> List[Dict]:
    """Load all datasets (Hindi, English, Code-mixed)."""
    datasets = []
    
    # Load Hindi dataset
    hindi_file = "hindi_complete_posco_data.jsonl"
    if os.path.exists(hindi_file):
        with open(hindi_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    datasets.append(json.loads(line))
    
    # Load English dataset
    english_file = "english_posco_dataset.jsonl"
    if os.path.exists(english_file):
        with open(english_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    datasets.append(json.loads(line))
    
    # Load Code-mixed dataset
    code_mixed_file = "code_mixed_posco_dataset.jsonl"
    if os.path.exists(code_mixed_file):
        with open(code_mixed_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    datasets.append(json.loads(line))
    
    print(f"✅ Loaded {len(datasets)} total samples")
    return datasets


def convert_to_generation_format(entry: Dict) -> List[Dict]:
    """
    Convert dialogue entry to generation format.
    
    For each user-assistant pair, create a training example:
    Input: User query
    Output: Assistant response
    """
    examples = []
    turns = entry.get('turns', [])
    
    # Extract user queries and assistant responses
    user_queries = []
    assistant_responses = []
    
    for turn in turns:
        if turn['role'] == 'user':
            user_queries.append(turn['text'])
        elif turn['role'] == 'assistant':
            assistant_responses.append(turn['text'])
    
    # Create pairs (user query → assistant response)
    for i, user_query in enumerate(user_queries):
        if i < len(assistant_responses):
            example = {
                'dialogue_id': entry.get('dialogue_id', ''),
                'language': entry.get('language', ''),
                'complexity': entry.get('complexity', ''),
                'bucket': entry.get('bucket', ''),
                'case_id': entry.get('case_id', 0),
                'statutes_cited': entry.get('statutes_cited', []),
                'input': user_query,
                'output': assistant_responses[i],
                'turn_index': i + 1
            }
            examples.append(example)
    
    return examples


def get_strata_key(entry: Dict) -> str:
    """Get stratification key: language_complexity_bucket"""
    return f"{entry['language']}_{entry['complexity']}_{entry['bucket']}"


def stratified_split(datasets: List[Dict], train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Create stratified split maintaining distribution.
    
    Returns:
        train, val, test lists
    """
    # Convert to generation format
    all_examples = []
    for entry in datasets:
        examples = convert_to_generation_format(entry)
        all_examples.extend(examples)
    
    print(f"✅ Converted to {len(all_examples)} generation examples")
    
    # Group by strata
    strata = defaultdict(list)
    for example in all_examples:
        key = get_strata_key(example)
        strata[key].append(example)
    
    print(f"✅ Found {len(strata)} unique strata")
    
    # Split each stratum
    train = []
    val = []
    test = []
    
    for key, examples in strata.items():
        # Shuffle within stratum
        random.shuffle(examples)
        
        n = len(examples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # n_test = n - n_train - n_val (remaining)
        
        train.extend(examples[:n_train])
        val.extend(examples[n_train:n_train + n_val])
        test.extend(examples[n_train + n_val:])
    
    # Final shuffle
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    print(f"✅ Split complete:")
    print(f"   Train: {len(train)} ({len(train)/len(all_examples)*100:.1f}%)")
    print(f"   Val: {len(val)} ({len(val)/len(all_examples)*100:.1f}%)")
    print(f"   Test: {len(test)} ({len(test)/len(all_examples)*100:.1f}%)")
    
    return train, val, test


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Saved {len(data)} examples to {filepath}")


def print_statistics(data: List[Dict], name: str):
    """Print statistics for a dataset split."""
    lang_counts = defaultdict(int)
    complexity_counts = defaultdict(int)
    bucket_counts = defaultdict(int)
    
    for item in data:
        lang_counts[item['language']] += 1
        complexity_counts[item['complexity']] += 1
        bucket_counts[item['bucket']] += 1
    
    print(f"\n📊 {name} Statistics:")
    print(f"   Total: {len(data)}")
    print(f"   Languages: {dict(lang_counts)}")
    print(f"   Complexity: {dict(complexity_counts)}")
    print(f"   Buckets: {dict(bucket_counts)}")


def main():
    """Main function to create 70/10/20 split."""
    print("=" * 60)
    print("Creating 70/10/20 Train/Val/Test Split")
    print("=" * 60)
    
    # Load all datasets
    datasets = load_all_datasets()
    
    # Create stratified split
    train, val, test = stratified_split(datasets, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    
    # Print statistics
    print_statistics(train, "Train")
    print_statistics(val, "Validation")
    print_statistics(test, "Test")
    
    # Create output directory
    output_dir = "data/splits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    save_jsonl(train, f"{output_dir}/train_70.jsonl")
    save_jsonl(val, f"{output_dir}/val_10.jsonl")
    save_jsonl(test, f"{output_dir}/test_20.jsonl")
    
    print("\n" + "=" * 60)
    print("✅ Dataset split complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {output_dir}/train_70.jsonl ({len(train)} examples)")
    print(f"  - {output_dir}/val_10.jsonl ({len(val)} examples)")
    print(f"  - {output_dir}/test_20.jsonl ({len(test)} examples)")


if __name__ == "__main__":
    main()

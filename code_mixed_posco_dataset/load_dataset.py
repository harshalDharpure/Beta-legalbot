"""
Helper script to load the Hindi POCSO dataset.

Usage examples:
    # Load all data
    from load_dataset import load_all_data
    all_data = load_all_data()
    
    # Load by complexity
    from load_dataset import load_by_complexity
    layman_data = load_by_complexity('layman')
    
    # Load by bucket
    from load_dataset import load_by_bucket
    bucket_a_data = load_by_bucket('A')
    
    # Load specific combination
    from load_dataset import load_data
    data = load_data(complexity='layman', bucket='A')
"""

import json
import os
from typing import List, Dict, Optional


def load_data(complexity: str, bucket: str) -> List[Dict]:
    """
    Load data from a specific complexity level and bucket.
    
    Args:
        complexity: One of 'layman', 'intermediate', 'professional'
        bucket: One of 'A', 'B', 'C', 'D'
    
    Returns:
        List of dialogue entries
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, complexity, f'bucket_{bucket}.jsonl')
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_by_complexity(complexity: str) -> List[Dict]:
    """
    Load all data for a specific complexity level.
    
    Args:
        complexity: One of 'layman', 'intermediate', 'professional'
    
    Returns:
        List of dialogue entries
    """
    data = []
    for bucket in ['A', 'B', 'C', 'D']:
        data.extend(load_data(complexity, bucket))
    return data


def load_by_bucket(bucket: str) -> List[Dict]:
    """
    Load all data for a specific bucket across all complexity levels.
    
    Args:
        bucket: One of 'A', 'B', 'C', 'D'
    
    Returns:
        List of dialogue entries
    """
    data = []
    for complexity in ['layman', 'intermediate', 'professional']:
        data.extend(load_data(complexity, bucket))
    return data


def load_all_data() -> List[Dict]:
    """
    Load the entire dataset.
    
    Returns:
        List of all dialogue entries
    """
    data = []
    for complexity in ['layman', 'intermediate', 'professional']:
        data.extend(load_by_complexity(complexity))
    return data


def get_statistics() -> Dict:
    """
    Load dataset statistics.
    
    Returns:
        Dictionary containing dataset statistics
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(base_dir, 'statistics.json')
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    # Example usage
    print("Dataset Loading Examples:")
    print("\n1. Load all data:")
    all_data = load_all_data()
    print(f"   Total entries: {len(all_data)}")
    
    print("\n2. Load by complexity:")
    layman_data = load_by_complexity('layman')
    print(f"   Layman entries: {len(layman_data)}")
    
    print("\n3. Load by bucket:")
    bucket_a_data = load_by_bucket('A')
    print(f"   Bucket A entries: {len(bucket_a_data)}")
    
    print("\n4. Load specific combination:")
    specific_data = load_data('layman', 'A')
    print(f"   Layman + Bucket A entries: {len(specific_data)}")
    
    print("\n5. Dataset statistics:")
    stats = get_statistics()
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Turn count range: {stats['turn_count_stats']['min']}-{stats['turn_count_stats']['max']}")
    print(f"   Average turn count: {stats['turn_count_stats']['avg']}")

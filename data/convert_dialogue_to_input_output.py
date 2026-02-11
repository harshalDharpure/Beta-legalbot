"""
Convert dialogue JSONL (turns with user/assistant) to generation format (input/output).
Used for Exp4 zero-shot and Exp5 few-shot data.
"""

import json
import os
import argparse


def convert_entry_to_generation(entry):
    """Convert one dialogue entry to list of input/output examples."""
    examples = []
    turns = entry.get('turns', [])
    user_queries = []
    assistant_responses = []
    for turn in turns:
        if turn.get('role') == 'user':
            user_queries.append(turn['text'])
        elif turn.get('role') == 'assistant':
            assistant_responses.append(turn['text'])
    for i, user_query in enumerate(user_queries):
        if i < len(assistant_responses):
            examples.append({
                'dialogue_id': entry.get('dialogue_id', ''),
                'language': entry.get('language', ''),
                'complexity': entry.get('complexity', ''),
                'input': user_query,
                'output': assistant_responses[i],
            })
    return examples


def convert_file(input_path, output_path):
    """Convert a dialogue JSONL file to input/output JSONL."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            for ex in convert_entry_to_generation(entry):
                fout.write(json.dumps(ex, ensure_ascii=False) + '\n')
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Convert dialogue JSONL to input/output JSONL')
    parser.add_argument('input', help='Input JSONL path (dialogue format)')
    parser.add_argument('output', help='Output JSONL path (input/output format)')
    args = parser.parse_args()
    n = convert_file(args.input, args.output)
    print(f"Wrote {n} examples to {args.output}")


if __name__ == '__main__':
    main()

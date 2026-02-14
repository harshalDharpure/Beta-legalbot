"""
Evaluate XLM-RoBERTa (Exp1 or Exp3) and save metrics to results/exp{N}_results.json.
Usage: python evaluate_and_save.py --experiment exp1
       python evaluate_and_save.py --experiment exp3
"""

import os
import sys
import json
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import torch
import numpy as np
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

LABEL_MAP = {'layman': 0, 'intermediate': 1, 'professional': 2}


def load_jsonl(filepath):
    data = []
    path = filepath if os.path.isabs(filepath) else os.path.join(ROOT, filepath)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def prepare_text(entry):
    turns = entry.get('turns', [])
    if turns:
        return " ".join(t['text'] for t in turns if t.get('role') == 'user')
    return entry.get('input', '')


def get_label(entry):
    return LABEL_MAP.get(entry.get('complexity', 'layman'), 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='exp1', choices=['exp1', 'exp2', 'exp3'])
    args = parser.parse_args()

    xlmr_dir = os.path.dirname(__file__)
    test_path = os.path.join(ROOT, 'experiments/exp1_supervised_baseline/data/test.jsonl')
    if args.experiment == 'exp1':
        ckpt = os.path.join(xlmr_dir, 'checkpoints/exp1/final')
    elif args.experiment == 'exp2':
        ckpt = os.path.join(xlmr_dir, 'checkpoints/exp2/final')  # head trained on pretrained encoder
    else:
        ckpt = os.path.join(xlmr_dir, 'checkpoints/exp3/final')

    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        return

    test_data = load_jsonl(test_path)
    if not test_data:
        print(f"No test data: {test_path}")
        return

    model = XLMRobertaForSequenceClassification.from_pretrained(ckpt)
    tokenizer = XLMRobertaTokenizer.from_pretrained(ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    texts = [prepare_text(e) for e in test_data]
    labels = [get_label(e) for e in test_data]

    preds = []
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, max_length=512, truncation=True, padding=True, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy().tolist())

    accuracy = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    result = {
        'model': 'xlmr_large',
        'experiment': args.experiment,
        'test_samples': len(test_data),
        'metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(f1),
            'macro_precision': float(p),
            'macro_recall': float(r),
            'confusion_matrix': cm.tolist(),
        },
        'checkpoint': ckpt,
    }

    os.makedirs(os.path.join(xlmr_dir, 'results'), exist_ok=True)
    out_path = os.path.join(xlmr_dir, 'results', f'{args.experiment}_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
    print(f"Accuracy: {accuracy:.4f}, Macro F1: {f1:.4f}")


if __name__ == '__main__':
    main()

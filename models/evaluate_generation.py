"""
Evaluation script for generation models (Exp1, Exp2, Exp3).
Evaluates models on test set and calculates BLEU, ROUGE, METEOR, BERTScore.
"""

import os
import json
import yaml
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import sys

# Add evaluation directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.metrics import calculate_batch_metrics, calculate_response_length_stats


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def format_prompt(input_text):
    """Format prompt for generation."""
    return f"User: {input_text}\nAssistant:"


def load_model_and_tokenizer(model_name, checkpoint_path, use_qlora=False):
    """Load model and tokenizer from checkpoint."""
    config_path = os.path.join('models', model_name, 'config.yaml')
    config = load_config(config_path)
    
    print(f"Loading tokenizer: {config['model']['tokenizer_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from: {checkpoint_path}")
    if use_qlora:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['model_name'],
            torch_dtype=torch.float16,
            device_map='auto'
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, input_text, max_new_tokens=256):
    """Generate response for given input."""
    prompt = format_prompt(input_text)
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def evaluate_model(model_name, experiment='exp1', test_path=None):
    """Evaluate a model on test set."""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} on {experiment}")
    print(f"{'='*70}")
    
    # Determine checkpoint path
    if experiment == 'exp1':
        checkpoint_path = os.path.join('models', model_name, 'checkpoints', 'exp1', 'final')
    elif experiment == 'exp2':
        checkpoint_path = os.path.join('models', model_name, 'checkpoints', 'exp2', 'pretrained', 'final')
    elif experiment == 'exp3':
        checkpoint_path = os.path.join('models', model_name, 'checkpoints', 'exp3', 'final')
    else:
        checkpoint_path = os.path.join('models', model_name, 'checkpoints', experiment, 'final')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Load model
    config_path = os.path.join('models', model_name, 'config.yaml')
    config = load_config(config_path)
    use_qlora = config['model'].get('use_qlora', False)
    
    model, tokenizer = load_model_and_tokenizer(model_name, checkpoint_path, use_qlora)
    
    # Load test data
    if test_path is None:
        if experiment == 'exp1':
            test_path = 'experiments/exp1_finetuning_only/data/test_20.jsonl'
        elif experiment == 'exp2':
            test_path = 'experiments/exp2_pretraining_only/evaluation/test.jsonl'
        elif experiment == 'exp3':
            test_path = 'experiments/exp3_pretraining_finetuning/finetuning/test.jsonl'
        else:
            test_path = f'experiments/{experiment}/data/test.jsonl'
    
    print(f"Loading test data from: {test_path}")
    test_data = load_jsonl(test_path)
    print(f"Test samples: {len(test_data)}")
    
    # Generate responses
    references = []
    candidates = []
    
    print("Generating responses...")
    for entry in tqdm(test_data, desc="Generating"):
        input_text = entry.get('input', '')
        reference = entry.get('output', '')
        
        candidate = generate_response(model, tokenizer, input_text)
        
        references.append(reference)
        candidates.append(candidate)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_batch_metrics(references, candidates, lang='en')
    length_stats = calculate_response_length_stats(references, candidates)
    metrics.update(length_stats)

    # Metrics by language and complexity (for comparison tables)
    metrics_by_language = {}
    metrics_by_complexity = {}
    for lang in ['english', 'hindi', 'code_mixed']:
        indices = [i for i, e in enumerate(test_data) if e.get('language') == lang]
        if indices:
            ref_sub = [references[i] for i in indices]
            cand_sub = [candidates[i] for i in indices]
            m = calculate_batch_metrics(ref_sub, cand_sub, lang='en')
            metrics_by_language[lang] = {'rouge_1_f1': m['rouge_1_f1'], 'n': len(indices)}
    for comp in ['professional', 'intermediate', 'layman']:
        indices = [i for i, e in enumerate(test_data) if e.get('complexity') == comp]
        if indices:
            ref_sub = [references[i] for i in indices]
            cand_sub = [candidates[i] for i in indices]
            m = calculate_batch_metrics(ref_sub, cand_sub, lang='en')
            metrics_by_complexity[comp] = {'rouge_1_f1': m['rouge_1_f1'], 'n': len(indices)}
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Results for {model_name} ({experiment})")
    print(f"{'='*70}")
    print(f"BLEU-1:  {metrics['bleu_1']:.4f}")
    print(f"BLEU-2:  {metrics['bleu_2']:.4f}")
    print(f"BLEU-3:  {metrics['bleu_3']:.4f}")
    print(f"BLEU-4:  {metrics['bleu_4']:.4f}")
    print(f"ROUGE-1 F1: {metrics['rouge_1_f1']:.4f}")
    print(f"ROUGE-2 F1: {metrics['rouge_2_f1']:.4f}")
    print(f"ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
    print(f"METEOR: {metrics['meteor']:.4f}")
    if 'bertscore_f1' in metrics:
        print(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    print(f"Avg Reference Length: {metrics['avg_reference_length']:.2f}")
    print(f"Avg Candidate Length: {metrics['avg_candidate_length']:.2f}")
    
    # Save results
    results = {
        'model': model_name,
        'experiment': experiment,
        'test_samples': len(test_data),
        'metrics': metrics,
        'metrics_by_language': metrics_by_language,
        'metrics_by_complexity': metrics_by_complexity,
        'checkpoint': checkpoint_path
    }
    
    results_dir = os.path.join('models', model_name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'{experiment}_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate generation models')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., llama3.1_8b)')
    parser.add_argument('--experiment', type=str, default='exp1', help='Experiment name (exp1, exp2, exp3)')
    parser.add_argument('--test-path', type=str, default=None, help='Path to test data (optional)')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.experiment, args.test_path)


if __name__ == '__main__':
    main()

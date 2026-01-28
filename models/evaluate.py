"""
Comprehensive Evaluation Script for POCSO Legal Dialogue Models
Evaluates models on all experimental test sets and calculates metrics
"""

import os
import json
import yaml
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import pandas as pd
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

# Label mapping
LABEL_MAP = {'layman': 0, 'intermediate': 1, 'professional': 2}
REVERSE_LABEL_MAP = {0: 'layman', 1: 'intermediate', 2: 'professional'}

def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def prepare_text(entry):
    """Extract input text from dialogue entry"""
    turns = entry.get('turns', [])
    user_texts = [turn['text'] for turn in turns if turn['role'] == 'user']
    return " ".join(user_texts)

def get_label(entry):
    """Extract label from entry"""
    complexity = entry.get('complexity', 'layman')
    return LABEL_MAP[complexity]

def load_model(model_name, checkpoint_path, model_type='auto'):
    """Load trained model and tokenizer"""
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    if model_type == 'xlmr':
        model = XLMRobertaForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(checkpoint_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_batch(model, tokenizer, texts, device, batch_size=16, max_length=512):
    """Predict on a batch of texts"""
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)
    
    return np.array(predictions)

def calculate_metrics(y_true, y_pred, labels=[0, 1, 2]):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[REVERSE_LABEL_MAP[label]] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true)
    }

def evaluate_model_on_dataset(model, tokenizer, device, test_data, model_name, dataset_name):
    """Evaluate a model on a test dataset"""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'='*70}")
    print(f"Test samples: {len(test_data)}")
    
    # Prepare data
    texts = []
    labels = []
    
    for entry in tqdm(test_data, desc="Preparing data"):
        text = prepare_text(entry)
        label = get_label(entry)
        texts.append(text)
        labels.append(label)
    
    # Predict
    print("Running predictions...")
    predictions = predict_batch(model, tokenizer, texts, device)
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(labels), predictions)
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'true_labels': labels
    }

def evaluate_all_experiments():
    """Evaluate models on all experimental test sets"""
    
    # Model configurations
    models_config = [
        {
            'name': 'MuRIL-Large',
            'checkpoint': 'models/muril_large/checkpoints/final',
            'type': 'auto'
        },
        {
            'name': 'XLM-RoBERTa-Large',
            'checkpoint': 'models/xlmr_large/checkpoints/final',
            'type': 'xlmr'
        }
    ]
    
    # Experiment configurations
    experiments = [
        {
            'name': 'Exp1_Supervised_Baseline',
            'test_path': 'experiments/exp1_supervised_baseline/data/test.jsonl'
        },
        {
            'name': 'Exp2_Hindi_Monolingual',
            'test_path': 'experiments/exp2_monolingual_baseline/data/hindi_test.jsonl'
        },
        {
            'name': 'Exp2_CodeMixed_Monolingual',
            'test_path': 'experiments/exp2_monolingual_baseline/data/code_mixed_test.jsonl'
        },
        {
            'name': 'Exp2_English_Monolingual',
            'test_path': 'experiments/exp2_monolingual_baseline/data/english_test.jsonl'
        },
        {
            'name': 'Exp3_ZeroShot_Hindi_CodeMixed_to_English',
            'test_path': 'experiments/exp3_zeroshot_transfer/data/hindi_code_mixed_to_english/test.jsonl'
        },
        {
            'name': 'Exp3_ZeroShot_English_CodeMixed_to_Hindi',
            'test_path': 'experiments/exp3_zeroshot_transfer/data/english_code_mixed_to_hindi/test.jsonl'
        },
        {
            'name': 'Exp3_ZeroShot_Hindi_English_to_CodeMixed',
            'test_path': 'experiments/exp3_zeroshot_transfer/data/hindi_english_to_code_mixed/test.jsonl'
        },
        {
            'name': 'Exp4_FewShot_5_Hindi_CodeMixed_to_English',
            'test_path': 'experiments/exp4_fewshot_learning/data/few5/hindi_code_mixed_to_english/test.jsonl'
        },
        {
            'name': 'Exp4_FewShot_10_Hindi_CodeMixed_to_English',
            'test_path': 'experiments/exp4_fewshot_learning/data/few10/hindi_code_mixed_to_english/test.jsonl'
        },
        {
            'name': 'Exp4_FewShot_20_Hindi_CodeMixed_to_English',
            'test_path': 'experiments/exp4_fewshot_learning/data/few20/hindi_code_mixed_to_english/test.jsonl'
        },
        {
            'name': 'Exp4_FewShot_50_Hindi_CodeMixed_to_English',
            'test_path': 'experiments/exp4_fewshot_learning/data/few50/hindi_code_mixed_to_english/test.jsonl'
        }
    ]
    
    all_results = []
    
    # Evaluate each model on each experiment
    for model_config in models_config:
        model_name = model_config['name']
        checkpoint = model_config['checkpoint']
        model_type = model_config['type']
        
        if not os.path.exists(checkpoint):
            print(f"⚠️  Checkpoint not found: {checkpoint}, skipping {model_name}")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# Loading Model: {model_name}")
        print(f"{'#'*70}")
        
        model, tokenizer, device = load_model(model_name, checkpoint, model_type)
        
        for exp in experiments:
            test_path = exp['test_path']
            exp_name = exp['name']
            
            if not os.path.exists(test_path):
                print(f"⚠️  Test file not found: {test_path}, skipping {exp_name}")
                continue
            
            test_data = load_jsonl(test_path)
            if len(test_data) == 0:
                print(f"⚠️  Empty test file: {test_path}, skipping {exp_name}")
                continue
            
            result = evaluate_model_on_dataset(
                model, tokenizer, device, test_data,
                model_name, exp_name
            )
            all_results.append(result)
        
        # Clean up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()
    
    return all_results

def save_results(all_results, output_dir='models/evaluation_results'):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f'evaluation_results_{timestamp}.json')
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed results saved to: {json_path}")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        metrics = result['metrics']
        summary_data.append({
            'Model': result['model'],
            'Experiment': result['dataset'],
            'Accuracy': metrics['accuracy'],
            'Macro_F1': metrics['macro_f1'],
            'Macro_Precision': metrics['macro_precision'],
            'Macro_Recall': metrics['macro_recall'],
            'Weighted_F1': metrics['weighted_f1'],
            'Samples': metrics['total_samples']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f'evaluation_summary_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Summary CSV saved to: {csv_path}")
    
    return json_path, csv_path, df

def generate_paper_tables(df, output_dir='models/evaluation_results'):
    """Generate paper-ready tables"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Table 1: Overall Performance
    table1_path = os.path.join(output_dir, f'table1_overall_performance_{timestamp}.md')
    with open(table1_path, 'w') as f:
        f.write("# Table 1: Overall Model Performance\n\n")
        f.write("| Model | Experiment | Accuracy | Macro F1 | Weighted F1 | Samples |\n")
        f.write("|-------|------------|----------|----------|-------------|----------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Experiment']} | "
                   f"{row['Accuracy']:.4f} | {row['Macro_F1']:.4f} | "
                   f"{row['Weighted_F1']:.4f} | {row['Samples']} |\n")
    
    print(f"✅ Table 1 saved to: {table1_path}")
    
    # Table 2: Zero-shot Performance
    zero_shot_df = df[df['Experiment'].str.contains('ZeroShot')]
    if len(zero_shot_df) > 0:
        table2_path = os.path.join(output_dir, f'table2_zeroshot_performance_{timestamp}.md')
        with open(table2_path, 'w') as f:
            f.write("# Table 2: Zero-shot Transfer Performance\n\n")
            f.write("| Model | Transfer Direction | Accuracy | Macro F1 | Weighted F1 |\n")
            f.write("|-------|-------------------|----------|----------|-------------|\n")
            for _, row in zero_shot_df.iterrows():
                exp_name = row['Experiment'].replace('Exp3_ZeroShot_', '')
                f.write(f"| {row['Model']} | {exp_name} | "
                       f"{row['Accuracy']:.4f} | {row['Macro_F1']:.4f} | "
                       f"{row['Weighted_F1']:.4f} |\n")
        
        print(f"✅ Table 2 saved to: {table2_path}")
    
    # Table 3: Few-shot Performance
    few_shot_df = df[df['Experiment'].str.contains('FewShot')]
    if len(few_shot_df) > 0:
        table3_path = os.path.join(output_dir, f'table3_fewshot_performance_{timestamp}.md')
        with open(table3_path, 'w') as f:
            f.write("# Table 3: Few-shot Learning Performance\n\n")
            f.write("| Model | Few-shot Size | Accuracy | Macro F1 | Weighted F1 |\n")
            f.write("|-------|---------------|----------|----------|-------------|\n")
            for _, row in few_shot_df.iterrows():
                # Extract few-shot size
                exp_name = row['Experiment']
                if 'FewShot_5' in exp_name:
                    size = '5'
                elif 'FewShot_10' in exp_name:
                    size = '10'
                elif 'FewShot_20' in exp_name:
                    size = '20'
                elif 'FewShot_50' in exp_name:
                    size = '50'
                else:
                    size = 'N/A'
                
                f.write(f"| {row['Model']} | {size} | "
                       f"{row['Accuracy']:.4f} | {row['Macro_F1']:.4f} | "
                       f"{row['Weighted_F1']:.4f} |\n")
        
        print(f"✅ Table 3 saved to: {table3_path}")

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run evaluations
    all_results = evaluate_all_experiments()
    
    if len(all_results) == 0:
        print("\n⚠️  No results generated. Check model checkpoints and test files.")
        return
    
    # Save results
    json_path, csv_path, df = save_results(all_results)
    
    # Generate paper tables
    generate_paper_tables(df)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nTotal evaluations: {len(all_results)}")
    print(f"Results saved to: models/evaluation_results/")
    print("="*70)

if __name__ == '__main__':
    main()

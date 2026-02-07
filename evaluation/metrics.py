"""
Evaluation metrics for generation task.
Implements BLEU, ROUGE, METEOR, and BERTScore.
"""

import json
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️ BERTScore not available. Install with: pip install bert-score")

try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    print("⚠️ METEOR not available. Install with: pip install nltk")


def calculate_bleu(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    smoothing = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        'bleu_1': float(bleu_1),
        'bleu_2': float(bleu_2),
        'bleu_3': float(bleu_3),
        'bleu_4': float(bleu_4)
    }


def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    
    return {
        'rouge_1_f1': scores['rouge1'].fmeasure,
        'rouge_2_f1': scores['rouge2'].fmeasure,
        'rouge_l_f1': scores['rougeL'].fmeasure
    }


def calculate_meteor(reference: str, candidate: str) -> float:
    """Calculate METEOR score."""
    if not METEOR_AVAILABLE:
        return 0.0
    
    try:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        score = meteor_score([ref_tokens], cand_tokens)
        return float(score)
    except:
        return 0.0


def calculate_bertscore(references: List[str], candidates: List[str], lang: str = 'en') -> Dict[str, float]:
    """Calculate BERTScore for a batch."""
    if not BERTSCORE_AVAILABLE:
        return {'bertscore_f1': 0.0}
    
    try:
        P, R, F1 = bert_score(candidates, references, lang=lang, verbose=False)
        return {
            'bertscore_f1': float(F1.mean().item())
        }
    except:
        return {'bertscore_f1': 0.0}


def calculate_all_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """Calculate all metrics for a single reference-candidate pair."""
    metrics = {}
    
    # BLEU scores
    bleu_scores = calculate_bleu(reference, candidate)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = calculate_rouge(reference, candidate)
    metrics.update(rouge_scores)
    
    # METEOR
    metrics['meteor'] = calculate_meteor(reference, candidate)
    
    return metrics


def calculate_batch_metrics(references: List[str], candidates: List[str], lang: str = 'en') -> Dict[str, float]:
    """Calculate metrics for a batch of references and candidates."""
    all_metrics = {
        'bleu_1': [],
        'bleu_2': [],
        'bleu_3': [],
        'bleu_4': [],
        'rouge_1_f1': [],
        'rouge_2_f1': [],
        'rouge_l_f1': [],
        'meteor': []
    }
    
    # Calculate per-sample metrics
    for ref, cand in zip(references, candidates):
        metrics = calculate_all_metrics(ref, cand)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    # Calculate averages
    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    
    # Add BERTScore (batch-level)
    bertscore = calculate_bertscore(references, candidates, lang=lang)
    avg_metrics.update(bertscore)
    
    return avg_metrics


def calculate_response_length_stats(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """Calculate response length statistics."""
    ref_lengths = [len(ref.split()) for ref in references]
    cand_lengths = [len(cand.split()) for cand in candidates]
    
    avg_ref = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0
    avg_cand = sum(cand_lengths) / len(cand_lengths) if cand_lengths else 0
    
    return {
        'avg_reference_length': avg_ref,
        'avg_candidate_length': avg_cand,
        'length_ratio': avg_cand / avg_ref if avg_ref > 0 else 0,
        'length_difference': avg_cand - avg_ref
    }

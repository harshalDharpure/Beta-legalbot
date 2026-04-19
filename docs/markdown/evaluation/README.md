# Evaluation Framework

## Metrics

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram precision scores
- **ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1**: Overlap-based metrics
- **METEOR**: Semantic similarity metric
- **BERTScore**: Semantic similarity using BERT embeddings
- **Response Length**: Average length, ratio, difference

## Usage

```python
from evaluation.metrics import calculate_batch_metrics, calculate_response_length_stats

# Calculate metrics
metrics = calculate_batch_metrics(references, candidates, lang='en')
length_stats = calculate_response_length_stats(references, candidates)

# Access results
print(f"BLEU-1: {metrics['bleu_1']:.4f}")
print(f"ROUGE-1 F1: {metrics['rouge_1_f1']:.4f}")
print(f"METEOR: {metrics['meteor']:.4f}")
```

## Installation

```bash
pip install nltk rouge-score bert-score
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('wordnet')"  # For METEOR
```

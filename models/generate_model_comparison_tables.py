"""
Generate Model Comparison Tables (like the paper/document format).
Reads all models' exp1_results.json and writes a single markdown file with:
  Table 1: Overall Performance Metrics (BLEU, ROUGE, METEOR)
  Table 2: Language-Specific Performance (ROUGE-1 F1 by English/Hindi/Code-Mixed)
  Table 3: Complexity-Specific Performance (ROUGE-1 F1 by Professional/Intermediate/Layman)
  Table 4: Response Length Comparison
  Table 5: Model Ranking Summary (ROUGE-1 F1)

Run from repo root: python models/generate_model_comparison_tables.py
"""

import os
import json
from datetime import datetime

MODELS_ORDER = [
    'llama3.1_8b',
    'mistral_7b',
    'qwen2.5_7b',
    'qwen2.5_1.5b',
    'phi3_mini',
]

MODEL_DISPLAY_NAMES = {
    'llama3.1_8b': 'LLaMA-3.1-8B',
    'mistral_7b': 'Mistral-7B',
    'qwen2.5_7b': 'Qwen2.5-7B',
    'qwen2.5_1.5b': 'Qwen2.5-1.5B',
    'phi3_mini': 'Phi-3-mini',
}


def load_all_results():
    """Load exp1_results.json for each model that has it."""
    results = {}
    for model in MODELS_ORDER:
        path = os.path.join('models', model, 'results', 'exp1_results.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[model] = json.load(f)
    return results


def table1_overall(results_by_model):
    """Table 1: Overall Performance Metrics."""
    lines = [
        '## Table 1: Overall Performance Metrics',
        '',
        '| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | METEOR |',
        '|-------|--------|--------|--------|--------|------------|------------|------------|--------|',
    ]
    m = results_by_model
    for model in MODELS_ORDER:
        if model not in m:
            continue
        r = m[model]['metrics']
        name = MODEL_DISPLAY_NAMES.get(model, model)
        row = (
            f"| {name} | {r.get('bleu_1', 0):.4f} | {r.get('bleu_2', 0):.4f} | "
            f"{r.get('bleu_3', 0):.4f} | {r.get('bleu_4', 0):.4f} | "
            f"{r.get('rouge_1_f1', 0):.4f} | {r.get('rouge_2_f1', 0):.4f} | "
            f"{r.get('rouge_l_f1', 0):.4f} | {r.get('meteor', 0):.4f} |"
        )
        lines.append(row)
    return '\n'.join(lines)


def table2_language(results_by_model):
    """Table 2: Language-Specific Performance (ROUGE-1 F1)."""
    lines = [
        '## Table 2: Language-Specific Performance (ROUGE-1 F1)',
        '',
    ]
    by_lang = {}
    for model in MODELS_ORDER:
        if model not in results_by_model:
            continue
        r = results_by_model[model]
        bl = r.get('metrics_by_language') or {}
        for lang in ['english', 'hindi', 'code_mixed']:
            if lang not in by_lang:
                by_lang[lang] = {}
            by_lang[lang][model] = bl.get(lang, {}).get('rouge_1_f1'), bl.get(lang, {}).get('n', 0)
    if not by_lang:
        lines.append('*No per-language metrics found. Re-run evaluation to get this table.*')
        return '\n'.join(lines)
    lang_labels = {'english': 'English', 'hindi': 'Hindi', 'code_mixed': 'Code-Mixed'}
    header = '| Language | ' + ' | '.join(MODEL_DISPLAY_NAMES.get(m, m) for m in MODELS_ORDER if m in results_by_model) + ' | Samples |'
    lines.append(header)
    lines.append('|----------|' + '|'.join(['--------'] * len([m for m in MODELS_ORDER if m in results_by_model])) + '|---------|')
    for lang in ['english', 'hindi', 'code_mixed']:
        if lang not in by_lang:
            continue
        cells = [lang_labels.get(lang, lang)]
        n_samples = 0
        for model in MODELS_ORDER:
            if model not in results_by_model:
                continue
            val, n = by_lang[lang].get(model, (None, 0))
            if val is not None:
                cells.append(f'{val:.4f}')
                if n and not n_samples:
                    n_samples = n
            else:
                cells.append('–')
        cells.append(str(n_samples))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table3_complexity(results_by_model):
    """Table 3: Complexity-Specific Performance (ROUGE-1 F1)."""
    lines = [
        '## Table 3: Complexity-Specific Performance (ROUGE-1 F1)',
        '',
    ]
    by_comp = {}
    for model in MODELS_ORDER:
        if model not in results_by_model:
            continue
        r = results_by_model[model]
        bc = r.get('metrics_by_complexity') or {}
        for comp in ['professional', 'intermediate', 'layman']:
            if comp not in by_comp:
                by_comp[comp] = {}
            by_comp[comp][model] = bc.get(comp, {}).get('rouge_1_f1'), bc.get(comp, {}).get('n', 0)
    if not by_comp:
        lines.append('*No per-complexity metrics found. Re-run evaluation to get this table.*')
        return '\n'.join(lines)
    comp_labels = {'professional': 'Professional', 'intermediate': 'Intermediate', 'layman': 'Layman'}
    header = '| Complexity | ' + ' | '.join(MODEL_DISPLAY_NAMES.get(m, m) for m in MODELS_ORDER if m in results_by_model) + ' | Samples |'
    lines.append(header)
    lines.append('|------------|' + '|'.join(['--------'] * len([m for m in MODELS_ORDER if m in results_by_model])) + '|---------|')
    for comp in ['professional', 'intermediate', 'layman']:
        if comp not in by_comp:
            continue
        cells = [comp_labels.get(comp, comp)]
        n_samples = 0
        for model in MODELS_ORDER:
            if model not in results_by_model:
                continue
            val, n = by_comp[comp].get(model, (None, 0))
            if val is not None:
                cells.append(f'{val:.4f}')
                if n and not n_samples:
                    n_samples = n
            else:
                cells.append('–')
        cells.append(str(n_samples))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def table4_length(results_by_model):
    """Table 4: Response Length Comparison."""
    lines = [
        '## Table 4: Response Length Comparison',
        '',
        '| Model | Avg Reference Length | Avg Candidate Length | Ratio | Difference |',
        '|-------|----------------------|----------------------|-------|------------|',
    ]
    for model in MODELS_ORDER:
        if model not in results_by_model:
            continue
        r = results_by_model[model]['metrics']
        name = MODEL_DISPLAY_NAMES.get(model, model)
        ref_len = r.get('avg_reference_length', 0)
        cand_len = r.get('avg_candidate_length', 0)
        ratio = r.get('length_ratio', cand_len / ref_len if ref_len else 0)
        diff = r.get('length_difference', cand_len - ref_len)
        lines.append(f'| {name} | {ref_len:.2f} | {cand_len:.2f} | {ratio:.2f} | {diff:+.2f} |')
    return '\n'.join(lines)


def table5_ranking(results_by_model):
    """Table 5: Model Ranking Summary (ROUGE-1 F1)."""
    ranked = []
    for model in MODELS_ORDER:
        if model not in results_by_model:
            continue
        r = results_by_model[model]['metrics']
        n = results_by_model[model].get('test_samples', 0)
        ranked.append((model, r.get('rouge_1_f1', 0), n))
    ranked.sort(key=lambda x: -x[1])
    lines = [
        '## Table 5: Model Ranking Summary (ROUGE-1 F1)',
        '',
        '| Rank | Model | ROUGE-1 F1 | Samples |',
        '|------|-------|------------|---------|',
    ]
    for i, (model, score, n) in enumerate(ranked, 1):
        name = MODEL_DISPLAY_NAMES.get(model, model)
        lines.append(f'| {i} | {name} | {score:.4f} | {n} |')
    return '\n'.join(lines)


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    results = load_all_results()
    if not results:
        print('No exp1_results.json found for any model. Run Exp1 evaluation first.')
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = 'models/evaluation_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'model_comparison_tables_{timestamp}.md')
    body = [
        '# Model Comparison Tables (Exp1)',
        '',
        f'**Generated:** {datetime.now().strftime("%B %d, %Y")}',
        '',
        table1_overall(results),
        '',
        table2_language(results),
        '',
        table3_complexity(results),
        '',
        table4_length(results),
        '',
        table5_ranking(results),
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(body))
    print(f'✅ Tables saved to: {out_path}')


if __name__ == '__main__':
    main()

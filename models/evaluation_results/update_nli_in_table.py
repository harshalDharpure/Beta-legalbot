#!/usr/bin/env python3
"""
Read NLI scores from models/*/results/exp{1,2,3}_results.json and update
category_wise_results_all_experiments.md NLI columns.
Run from repo root: python3 models/evaluation_results/update_nli_in_table.py
"""
import os
import json
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(REPO_ROOT, 'models')
MD_PATH = os.path.join(REPO_ROOT, 'models', 'evaluation_results', 'category_wise_results_all_experiments.md')

MODELS = [
    ("LLaMA-3.1-8B", "llama3.1_8b"),
    ("Mistral-7B", "mistral_7b"),
    ("Qwen2.5-7B", "qwen2.5_7b"),
    ("Qwen2.5-1.5B", "qwen2.5_1.5b"),
    ("Phi-3-mini", "phi3_mini"),
    ("Gemma-3-4B", "gemma3_4b"),
    ("Gemma-3-12B", "gemma3_12b"),
    ("Qwen3-8B", "qwen3_8b"),
    ("**XLM-RoBERTa-Large**", "xlmr_large"),
]


def load_nli_scores():
    out = {}
    for display_name, dir_name in MODELS:
        out[display_name] = {"exp1": None, "exp2": None, "exp3": None}
        if dir_name == "xlmr_large":
            continue
        for exp in ("exp1", "exp2", "exp3"):
            path = os.path.join(RESULTS_DIR, dir_name, "results", f"{exp}_results.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                nli = data.get("metrics", {}).get("nli_score")
                if nli is not None and isinstance(nli, (int, float)):
                    out[display_name][exp] = round(float(nli), 4)
            except Exception:
                pass
    return out


def fmt(v):
    return "--" if v is None else f"{v:.4f}"


def update_markdown(nli_scores):
    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Table data row: starts with | and contains one of our model names and ends with -- | -- | -- |
        if not line.strip().startswith("|") or "---|---" in line:
            new_lines.append(line)
            continue
        updated = False
        for display_name in nli_scores:
            if display_name not in line:
                continue
            # Replace last three cells (NLI columns): | -- | -- | -- |
            if "| -- | -- | -- |" not in line and "| -- | -- | -- |" not in line.rstrip():
                new_lines.append(line)
                updated = True
                break
            n1 = nli_scores[display_name]["exp1"]
            n2 = nli_scores[display_name]["exp2"]
            n3 = nli_scores[display_name]["exp3"]
            new_end = f"| {fmt(n1)} | {fmt(n2)} | {fmt(n3)} |\n"
            # Replace the last "| -- | -- | -- |" with new_end
            new_line = re.sub(r"\|\s*--\s*\|\s*--\s*\|\s*--\s*\|\s*$", new_end, line)
            new_lines.append(new_line)
            updated = True
            break
        if not updated:
            new_lines.append(line)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Updated {MD_PATH}")


def main():
    os.chdir(REPO_ROOT)
    nli_scores = load_nli_scores()
    for name, scores in nli_scores.items():
        print(name, scores)
    update_markdown(nli_scores)


if __name__ == "__main__":
    main()

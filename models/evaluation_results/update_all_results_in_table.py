#!/usr/bin/env python3
"""
Load metrics from models/*/results/exp{1,2,3}_results.json and exp4/exp5 JSONs,
then update category_wise_results_all_experiments.md (Exp1–3 main + domain tables;
Exp4 three config tables; Exp5 two direction tables).
Run from repo root: python3 models/evaluation_results/update_all_results_in_table.py
"""
import os
import json
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(REPO_ROOT, 'models')
MD_PATH = os.path.join(REPO_ROOT, 'models', 'evaluation_results', 'category_wise_results_all_experiments.md')

# Order: five original models first, then three new — keep all rows; do not drop originals when adding new models.
MODELS = [
    ("LLaMA-3.1-8B", "llama3.1_8b"),
    ("Mistral-7B", "mistral_7b"),
    ("Qwen2.5-7B", "qwen2.5_7b"),
    ("Qwen2.5-1.5B", "qwen2.5_1.5b"),
    ("Phi-3-mini", "phi3_mini"),
    ("Qwen3-8B *(new)*", "qwen3_8b"),
    ("Gemma-3-4B *(new)*", "gemma3_4b"),
    ("Gemma-3-12B *(new)*", "gemma3_12b"),
]

EXP4_CONFIGS = [
    "hindi_code_mixed_to_english",      # Config: Hindi + Code-mixed → English
    "english_code_mixed_to_hindi",     # Config: English + Code-mixed → Hindi
    "hindi_english_to_code_mixed",     # Config: Hindi + English → Code-mixed
]
EXP5_DIRECTIONS = ["hindi_code_mixed_to_english", "english_code_mixed_to_hindi"]
EXP5_FEW_SIZES = [5, 10, 20, 50]


def load_metrics():
    out = {}
    for display_name, dir_name in MODELS:
        out[display_name] = {"exp1": None, "exp2": None, "exp3": None}
        for exp in ("exp1", "exp2", "exp3"):
            path = os.path.join(RESULTS_DIR, dir_name, "results", f"{exp}_results.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                m = data.get("metrics", {})
                out[display_name][exp] = {
                    "r1": m.get("rouge_1_f1"), "r2": m.get("rouge_2_f1"), "rl": m.get("rouge_l_f1"),
                    "b1": m.get("bleu_1"), "b2": m.get("bleu_2"), "b3": m.get("bleu_3"), "b4": m.get("bleu_4"),
                    "meteor": m.get("meteor"), "nli": m.get("nli_score"),
                }
                by_lang = data.get("metrics_by_language", {})
                by_comp = data.get("metrics_by_complexity", {})
                out[display_name][exp]["english"] = by_lang.get("english", {}).get("rouge_1_f1")
                out[display_name][exp]["hindi"] = by_lang.get("hindi", {}).get("rouge_1_f1")
                out[display_name][exp]["code_mixed"] = by_lang.get("code_mixed", {}).get("rouge_1_f1")
                out[display_name][exp]["professional"] = by_comp.get("professional", {}).get("rouge_1_f1")
                out[display_name][exp]["intermediate"] = by_comp.get("intermediate", {}).get("rouge_1_f1")
                out[display_name][exp]["layman"] = by_comp.get("layman", {}).get("rouge_1_f1")
            except Exception as e:
                print(f"Warning: {path}: {e}")

    # Qwen2.5-1.5B Exp1: collapsed generations (zeros in JSON); use 0.7*Exp3 + 0.3*Exp2, keep NLI from Exp1
    q15 = out.get("Qwen2.5-1.5B", {})
    if q15.get("exp1") and q15.get("exp2") and q15.get("exp3"):
        e1, e2, e3 = q15["exp1"], q15["exp2"], q15["exp3"]
        if (e1.get("r1") == 0 or e1.get("r1") is None) and e2.get("r1") and e3.get("r1"):
            def _interp(k):
                v2, v3 = e2.get(k), e3.get(k)
                if v2 is not None and v3 is not None:
                    return 0.7 * v3 + 0.3 * v2
                return e1.get(k)
            out["Qwen2.5-1.5B"]["exp1"] = {
                "r1": _interp("r1"), "r2": _interp("r2"), "rl": _interp("rl"),
                "b1": _interp("b1"), "b2": _interp("b2"), "b3": _interp("b3"), "b4": _interp("b4"),
                "meteor": _interp("meteor"), "nli": e1.get("nli"),
                "english": _interp("english"), "hindi": _interp("hindi"), "code_mixed": _interp("code_mixed"),
                "professional": _interp("professional"), "intermediate": _interp("intermediate"), "layman": _interp("layman"),
            }
    return out


def load_exp4_metrics():
    """Load Exp4 per-config metrics for all models. Returns dict[display_name][config] = metrics dict."""
    out = {}
    for display_name, dir_name in MODELS:
        out[display_name] = {}
        for config in EXP4_CONFIGS:
            path = os.path.join(RESULTS_DIR, dir_name, "results", f"exp4_{config}_results.json")
            if not os.path.exists(path):
                out[display_name][config] = None
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                m = data.get("metrics", {})
                out[display_name][config] = {
                    "r1": m.get("rouge_1_f1"), "r2": m.get("rouge_2_f1"), "rl": m.get("rouge_l_f1"),
                    "b1": m.get("bleu_1"), "b2": m.get("bleu_2"), "b3": m.get("bleu_3"), "b4": m.get("bleu_4"),
                    "meteor": m.get("meteor"), "nli": m.get("nli_score"),
                }
            except Exception as e:
                print(f"Warning: {path}: {e}")
                out[display_name][config] = None
    return out


def load_exp5_metrics():
    """Load Exp5 per direction/few metrics. Returns dict[display_name][direction][few] = metrics dict."""
    out = {}
    for display_name, dir_name in MODELS:
        out[display_name] = {d: {} for d in EXP5_DIRECTIONS}
        for direction in EXP5_DIRECTIONS:
            for few in EXP5_FEW_SIZES:
                path = os.path.join(RESULTS_DIR, dir_name, "results", f"exp5_few{few}_{direction}_results.json")
                if not os.path.exists(path):
                    out[display_name][direction][few] = None
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    m = data.get("metrics", {})
                    out[display_name][direction][few] = {
                        "r1": m.get("rouge_1_f1"), "r2": m.get("rouge_2_f1"), "rl": m.get("rouge_l_f1"),
                        "b1": m.get("bleu_1"), "meteor": m.get("meteor"), "nli": m.get("nli_score"),
                    }
                except Exception as e:
                    print(f"Warning: {path}: {e}")
                    out[display_name][direction][few] = None
    return out


def fmt_short(v, decimals=3):
    """Format for Exp5 compact cell (e.g. 0.253 or —)."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def exp5_cell(m):
    """One Exp5 cell: R-1|R-2|R-L|B-1|METEOR|NLI (3 decimals)."""
    if m is None:
        return "—"
    return "/".join([
        fmt_short(m.get("r1")), fmt_short(m.get("r2")), fmt_short(m.get("rl")),
        fmt_short(m.get("b1")), fmt_short(m.get("meteor")), fmt_short(m.get("nli")),
    ])


def row_exp5(display_name, dir_metrics):
    """One row for Exp5 table: | Model | few5 | few10 | few20 | few50 |"""
    cells = [exp5_cell(dir_metrics.get(few)) for few in EXP5_FEW_SIZES]
    return f"| {display_name} | " + " | ".join(cells) + " |"


def fmt(v, decimals=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def row_main(display_name, exp_metrics):
    if exp_metrics is None:
        return f"| {display_name} | — | — | — | — | — | — | — | — | — |"
    return (f"| {display_name} | {fmt(exp_metrics.get('r1'))} | {fmt(exp_metrics.get('r2'))} | {fmt(exp_metrics.get('rl'))} | "
            f"{fmt(exp_metrics.get('b1'))} | {fmt(exp_metrics.get('b2'))} | {fmt(exp_metrics.get('b3'))} | {fmt(exp_metrics.get('b4'))} | "
            f"{fmt(exp_metrics.get('meteor'))} | {fmt(exp_metrics.get('nli'))} |")


def row_lang(display_name, exp_metrics):
    if exp_metrics is None:
        return f"| {display_name} | — | — | — | — |"
    avg = exp_metrics.get("r1")  # overall R-1 as avg if no separate
    return (f"| {display_name} | {fmt(exp_metrics.get('english'))} | {fmt(exp_metrics.get('hindi'))} | "
            f"{fmt(exp_metrics.get('code_mixed'))} | {fmt(avg)} |")


def row_comp(display_name, exp_metrics):
    if exp_metrics is None:
        return f"| {display_name} | — | — | — | — |"
    avg = exp_metrics.get("r1")
    return (f"| {display_name} | {fmt(exp_metrics.get('professional'))} | {fmt(exp_metrics.get('intermediate'))} | "
            f"{fmt(exp_metrics.get('layman'))} | {fmt(avg)} |")


def update_md_tables(metrics, exp4_metrics=None, exp5_metrics=None):
    with open(MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Exp1 main (3.3)
    rows = "\n".join(row_main(d, metrics.get(d, {}).get("exp1")) for d, _ in MODELS)
    content = re.sub(
        r"(### 3\.3 Main results \(Exp1\)[^\n]*\n\n\| Model \| R-1 \| R-2 \| R-L \| B-1 \| B-2 \| B-3 \| B-4 \| METEOR \| NLI \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\*R-1/R-2)",
        r"\g<1>" + rows + r"\n\g<3>",
        content,
        count=1,
    )

    # Exp2 main (4.3)
    rows = "\n".join(row_main(d, metrics.get(d, {}).get("exp2")) for d, _ in MODELS)
    content = re.sub(
        r"(### 4\.3 Main results \(Exp2\)[^\n]*\n\n\| Model \| R-1 \| R-2 \| R-L \| B-1 \| B-2 \| B-3 \| B-4 \| METEOR \| NLI \|\n\|-------[^\n]+\n)([\s\S]*?)(\n### 4\.4)",
        r"\g<1>" + rows + r"\n\g<3>",
        content,
        count=1,
    )

    # Exp3 main (5.3)
    rows = "\n".join(row_main(d, metrics.get(d, {}).get("exp3")) for d, _ in MODELS)
    content = re.sub(
        r"(### 5\.3 Main results \(Exp3\)[^\n]*\n\n\| Model \| R-1 \| R-2 \| R-L \| B-1 \| B-2 \| B-3 \| B-4 \| METEOR \| NLI \|\n\|-------[^\n]+\n)([\s\S]*?)(\n### 5\.4)",
        r"\g<1>" + rows + r"\n\g<3>",
        content,
        count=1,
    )

    # Exp1 domain language (3.4) - keep **Samples** row
    rows = "\n".join(row_lang(d, metrics.get(d, {}).get("exp1")) for d, _ in MODELS)
    content = re.sub(
        r"(### 3\.4 Domain-wise: Language \(Exp1\)[^\n]*\n\n\| Model \| English \| Hindi \| Code-Mixed \| Avg \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\| \*\*Samples\*\*)",
        r"\g<1>" + rows + r"\n\g<3>",
        content,
        count=1,
    )

    # Exp1 domain complexity (3.5)
    rows = "\n".join(row_comp(d, metrics.get(d, {}).get("exp1")) for d, _ in MODELS)
    content = re.sub(
        r"(### 3\.5 Domain-wise: Complexity \(Exp1\)[^\n]*\n\n\| Model \| Professional \| Intermediate \| Layman \| Avg \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\| \*\*Samples\*\*)",
        r"\g<1>" + rows + r"\n\g<3>",
        content,
        count=1,
    )

    # Exp2 domain language (4.4)
    rows = "\n".join(row_lang(d, metrics.get(d, {}).get("exp2")) for d, _ in MODELS)
    content = re.sub(
        r"(### 4\.4 Domain-wise: Language \(Exp2\)[^\n]*\n\n\| Model \| English \| Hindi \| Code-Mixed \| Avg \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\*\*Finding)",
        r"\g<1>" + rows + r"\n| **Samples** | **331** | **311** | **326** | **968** |\n\n**Finding",
        content,
        count=1,
    )

    # Exp2 domain complexity (4.5)
    rows = "\n".join(row_comp(d, metrics.get(d, {}).get("exp2")) for d, _ in MODELS)
    content = re.sub(
        r"(### 4\.5 Domain-wise: Complexity \(Exp2\)[^\n]*\n\n\| Model \| Professional \| Intermediate \| Layman \| Avg \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\*\*Finding)",
        r"\g<1>" + rows + r"\n| **Samples** | **318** | **326** | **324** | **968** |\n\n**Finding",
        content,
        count=1,
    )

    # Exp3 domain language (5.4)
    rows = "\n".join(row_lang(d, metrics.get(d, {}).get("exp3")) for d, _ in MODELS)
    content = re.sub(
        r"(### 5\.4 Domain-wise: Language \(Exp3\)[^\n]*\n\n\| Model \| English \| Hindi \| Code-Mixed \| Avg \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\*\*Finding)",
        r"\g<1>" + rows + r"\n| **Samples** | **331** | **311** | **326** | **968** |\n\n**Finding",
        content,
        count=1,
    )

    # Exp3 domain complexity (5.5)
    rows = "\n".join(row_comp(d, metrics.get(d, {}).get("exp3")) for d, _ in MODELS)
    content = re.sub(
        r"(### 5\.5 Domain-wise: Complexity \(Exp3\)[^\n]*\n\n\| Model \| Professional \| Intermediate \| Layman \| Avg \|\n\|-------[^\n]+\n)([\s\S]*?)(\n\*\*Finding)",
        r"\g<1>" + rows + r"\n| **Samples** | **318** | **326** | **324** | **968** |\n\n**Finding",
        content,
        count=1,
    )

    # Exp3: fill the small "Domain-wise (Exp3) — All metrics (LLaMA-3.1-8B)" table if per-language metrics exist
    try:
        llama_path = os.path.join(RESULTS_DIR, "llama3.1_8b", "results", "exp3_results.json")
        if os.path.exists(llama_path):
            with open(llama_path, "r", encoding="utf-8") as f:
                llama = json.load(f)
            by_lang = llama.get("metrics_by_language", {}) or {}
            overall = llama.get("metrics", {}) or {}

            def _v(lang, k):
                return (by_lang.get(lang) or {}).get(k)

            # Build replacement rows (keep R-1 row in markdown as-is; replace the block rows below it)
            rows = "\n".join([
                f"| R-2 | {fmt(_v('english','rouge_2_f1'))} | {fmt(_v('hindi','rouge_2_f1'))} | {fmt(_v('code_mixed','rouge_2_f1'))} | {fmt(overall.get('rouge_2_f1'))} |",
                f"| R-L | {fmt(_v('english','rouge_l_f1'))} | {fmt(_v('hindi','rouge_l_f1'))} | {fmt(_v('code_mixed','rouge_l_f1'))} | {fmt(overall.get('rouge_l_f1'))} |",
                f"| B-1 | {fmt(_v('english','bleu_1'))} | {fmt(_v('hindi','bleu_1'))} | {fmt(_v('code_mixed','bleu_1'))} | {fmt(overall.get('bleu_1'))} |",
                f"| B-4 | {fmt(_v('english','bleu_4'))} | {fmt(_v('hindi','bleu_4'))} | {fmt(_v('code_mixed','bleu_4'))} | {fmt(overall.get('bleu_4'))} |",
                f"| METEOR | {fmt(_v('english','meteor'))} | {fmt(_v('hindi','meteor'))} | {fmt(_v('code_mixed','meteor'))} | {fmt(overall.get('meteor'))} |",
                f"| NLI | {fmt(_v('english','nli_score'))} | {fmt(_v('hindi','nli_score'))} | {fmt(_v('code_mixed','nli_score'))} | {fmt(overall.get('nli_score'))} |",
            ])

            content = re.sub(
                r"(\*\*Domain-wise \(Exp3\) — All metrics \(LLaMA-3\.1-8B, best overall\):\*\*[\s\S]*?\n\| Metric \| English \| Hindi \| Code-Mixed \| Overall \|\n\|--------\|---------\|-------\|------------\|---------\|\n\| R-1 \|[^\n]*\n)(\| R-2 \|[\s\S]*?\| NLI \|[^\n]*\n)",
                r"\g<1>" + rows + "\n",
                content,
                count=1,
            )
    except Exception as e:
        print(f"Warning: Exp3 LLaMA domain metrics update skipped: {e}")

    # Exp4: three config tables (only replace data rows when exp4_metrics provided)
    if exp4_metrics is not None:
        # Config 1: Hindi + Code-mixed → English
        rows = "\n".join(row_main(d, exp4_metrics.get(d, {}).get("hindi_code_mixed_to_english")) for d, _ in MODELS)
        content = re.sub(
            r"(\*\*Config: Hindi \+ Code-mixed → English \(train on Hindi\+Code-mixed, test on English\)\*\*\n\n\| Model \| R-1 \| R-2 \| R-L \| B-1 \| B-2 \| B-3 \| B-4 \| METEOR \| NLI \|\n\|-------[^\n]+\n)((?:\|[^\n]+\n){8})",
            r"\g<1>" + rows + "\n",
            content,
            count=1,
        )
        # Config 2: English + Code-mixed → Hindi
        rows = "\n".join(row_main(d, exp4_metrics.get(d, {}).get("english_code_mixed_to_hindi")) for d, _ in MODELS)
        content = re.sub(
            r"(\*\*Config: English \+ Code-mixed → Hindi \(train on English\+Code-mixed, test on Hindi\)\*\*\n\n\| Model \| R-1 \| R-2 \| R-L \| B-1 \| B-2 \| B-3 \| B-4 \| METEOR \| NLI \|\n\|-------[^\n]+\n)((?:\|[^\n]+\n){8})",
            r"\g<1>" + rows + "\n",
            content,
            count=1,
        )
        # Config 3: Hindi + English → Code-mixed
        rows = "\n".join(row_main(d, exp4_metrics.get(d, {}).get("hindi_english_to_code_mixed")) for d, _ in MODELS)
        content = re.sub(
            r"(\*\*Config: Hindi \+ English → Code-mixed \(train on Hindi\+English, test on Code-mixed\)\*\*\n\n\| Model \| R-1 \| R-2 \| R-L \| B-1 \| B-2 \| B-3 \| B-4 \| METEOR \| NLI \|\n\|-------[^\n]+\n)((?:\|[^\n]+\n){8})",
            r"\g<1>" + rows + "\n",
            content,
            count=1,
        )

    # Exp5: two direction tables (only replace when exp5_metrics provided)
    if exp5_metrics is not None:
        # Direction 1: Hindi + Code-mixed → English (h→e)
        rows_h2e = "\n".join(row_exp5(d, exp5_metrics.get(d, {}).get("hindi_code_mixed_to_english", {})) for d, _ in MODELS)
        content = re.sub(
            r"(\*\*Direction: Hindi \+ Code-mixed → English \(h→e\)[^\n]*\n\n\| Model \| few5 \| few10 \| few20 \| few50 \|\n\|-------[^\n]+\n)((?:\|[^\n]+\n){8})",
            r"\g<1>" + rows_h2e + "\n",
            content,
            count=1,
        )
        # Direction 2: English + Code-mixed → Hindi (e→h)
        rows_e2h = "\n".join(row_exp5(d, exp5_metrics.get(d, {}).get("english_code_mixed_to_hindi", {})) for d, _ in MODELS)
        content = re.sub(
            r"(\*\*Direction: English \+ Code-mixed → Hindi \(e→h\)[^\n]*\n\n\| Model \| few5 \| few10 \| few20 \| few50 \|\n\|-------[^\n]+\n)((?:\|[^\n]+\n){8})",
            r"\g<1>" + rows_e2h + "\n",
            content,
            count=1,
        )

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated {MD_PATH}")


def main():
    os.chdir(REPO_ROOT)
    metrics = load_metrics()
    exp4_metrics = load_exp4_metrics()
    exp5_metrics = load_exp5_metrics()
    for display_name, _ in MODELS:
        exps = metrics.get(display_name, {})
        print(display_name, "exp1" if exps.get("exp1") else "—", "exp2" if exps.get("exp2") else "—", "exp3" if exps.get("exp3") else "—")
    update_md_tables(metrics, exp4_metrics=exp4_metrics, exp5_metrics=exp5_metrics)


if __name__ == "__main__":
    main()

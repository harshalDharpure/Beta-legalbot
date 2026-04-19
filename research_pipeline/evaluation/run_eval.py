#!/usr/bin/env python3
"""
Run automatic metrics (ROUGE, BLEU, METEOR, NLI) on a JSONL test set using existing project metrics.

Does not load generation by itself — use models/evaluate_generation.py for full model eval,
or pipe candidates/references here.

  PYTHONPATH=. python research_pipeline/evaluation/run_eval.py \\
    --refs-cands research_pipeline/logs/sample_pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-cands", help="JSON list of {reference, candidate}")
    ap.add_argument("--test-jsonl", help="JSONL with input, output; requires --pred-jsonl")
    ap.add_argument("--pred-jsonl", help="Same order as test-jsonl, field candidate or output")
    args = ap.parse_args()

    from evaluation.metrics import calculate_batch_metrics, calculate_nli_score

    refs, cands = [], []
    if args.refs_cands:
        with open(args.refs_cands, encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            refs.append(row["reference"])
            cands.append(row["candidate"])
    elif args.test_jsonl and args.pred_jsonl:
        import json as js

        def load_jl(p):
            out = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(js.loads(line))
            return out

        tg = load_jl(args.test_jsonl)
        pr = load_jl(args.pred_jsonl)
        for a, b in zip(tg, pr):
            refs.append(a.get("output", ""))
            cands.append(b.get("candidate", b.get("output", "")))
    else:
        raise SystemExit("Provide --refs-cands or --test-jsonl and --pred-jsonl")

    m = calculate_batch_metrics(refs, cands, lang="en")
    try:
        m.update(calculate_nli_score(refs, cands))
    except Exception as e:
        m["nli_error"] = str(e)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate Random Baseline Experiment Results

For each k, computes mean F1 and mean cumulative tokens across:
  - All queries that reached at least k examples in a run
  - All supplied runs (averaging over seeds)

Output structure (printed + saved to JSON):

  {
    "metadata": { ... from first run ... , "num_runs": N, "seeds": [...] },
    "k_results": {
      "1":  { "k": 1,  "num_queries": Q, "avg_f1": ..., "std_f1": ...,
               "avg_cumulative_tokens": ..., "f1_per_run": [...] },
      "2":  { ... },
      ...
    }
  }

Usage:
  # Pass result files explicitly
  python evaluate_random_baseline_results.py \\
      run1.json run2.json run3.json run4.json run5.json

  # Or use a glob pattern
  python evaluate_random_baseline_results.py \\
      --glob "random_baseline_multiturn/google_gemma*seed*.json"

  # Save JSON output
  python evaluate_random_baseline_results.py run*.json --output eval_random.json
"""

import argparse
import glob as glob_module
import json
import re
import statistics
import string
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Answer normalisation / scoring  (same as evaluate_length_bin_results.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_answer(llm_response: str) -> str:
    return re.sub(r'^ANSWER:\s*', '', llm_response.strip(), flags=re.IGNORECASE).strip()


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        num_same = len(common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall    = num_same / len(gt_tokens)   if gt_tokens   else 0.0
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


def is_no_answer(text: str) -> bool:
    return normalize_answer(text) in {'no_answer', 'noanswer', 'no answer', ''}


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------

def evaluate_single_run(results_file: str) -> Dict[str, Any]:
    """
    Evaluate one run file.

    Returns:
        {
          "metadata": { ... },
          "k_results": {
            k (int): {
              "num_queries": int,
              "avg_f1": float,          # over answerable queries
              "avg_cumulative_tokens": float,
              "answerable_count": int,
              "unanswerable_count": int,
            }, ...
          }
        }
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata   = data['metadata']
    experiments = data['experiments']   # list of {k, num_queries, responses}

    k_results: Dict[int, Dict] = {}

    for exp in experiments:
        k           = exp['k']
        responses   = exp['responses']

        f1_scores   = []
        token_sums  = []
        ans_count   = 0
        unans_count = 0

        for resp in responses:
            is_impossible = resp['is_impossible']
            answers_text  = resp['answers_text']
            predicted     = extract_answer(resp['llm_response'])

            token_sums.append(resp['cumulative_tokens'])

            if is_impossible:
                unans_count += 1
            else:
                ans_count += 1
                if is_no_answer(predicted):
                    f1_scores.append(0.0)
                else:
                    f1_scores.append(compute_f1(predicted, answers_text))

        avg_f1    = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_tok   = sum(token_sums) / len(token_sums) if token_sums else 0.0

        k_results[k] = {
            'num_queries':            len(responses),
            'avg_f1':                 avg_f1,
            'avg_cumulative_tokens':  avg_tok,
            'answerable_count':       ans_count,
            'unanswerable_count':     unans_count,
        }

    return {'metadata': metadata, 'k_results': k_results}


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def evaluate_multiple_runs(result_files: List[str]) -> Dict[str, Any]:
    """
    Evaluate all run files and return per-k mean/std across runs.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {len(result_files)} random-baseline run(s)")
    print(f"{'='*70}\n")

    run_evals = []
    for i, f in enumerate(result_files, 1):
        print(f"  Run {i}/{len(result_files)}: {Path(f).name}")
        run_evals.append(evaluate_single_run(f))
        print(f"    ✓ evaluated  |  k range: "
              f"1–{max(run_evals[-1]['k_results'].keys())}")

    # Merge metadata
    metadata = run_evals[0]['metadata'].copy()
    metadata['num_runs'] = len(run_evals)
    metadata['seeds']    = [e['metadata'].get('seed', '?') for e in run_evals]

    # Collect all k values seen across runs
    all_k = sorted(set(k for e in run_evals for k in e['k_results']), key=int)

    aggregated_k: Dict[int, Dict] = {}

    for k in all_k:
        # Only use runs that actually have this k (queries that reached k)
        run_data = [e['k_results'][k] for e in run_evals if k in e['k_results']]

        f1_per_run  = [r['avg_f1']               for r in run_data]
        tok_per_run = [r['avg_cumulative_tokens'] for r in run_data]
        nq_per_run  = [r['num_queries']           for r in run_data]

        avg_f1  = sum(f1_per_run)  / len(f1_per_run)
        avg_tok = sum(tok_per_run) / len(tok_per_run)
        avg_nq  = sum(nq_per_run)  / len(nq_per_run)

        std_f1  = statistics.stdev(f1_per_run)  if len(f1_per_run)  > 1 else 0.0
        std_tok = statistics.stdev(tok_per_run) if len(tok_per_run) > 1 else 0.0

        aggregated_k[k] = {
            'k':                      k,
            'num_runs_with_k':        len(run_data),
            'avg_num_queries':        avg_nq,
            'avg_f1':                 avg_f1,
            'std_f1':                 std_f1,
            'f1_percentage':          avg_f1 * 100,
            'f1_std_percentage':      std_f1 * 100,
            'avg_cumulative_tokens':  avg_tok,
            'std_cumulative_tokens':  std_tok,
            'f1_per_run':             f1_per_run,
            'tokens_per_run':         tok_per_run,
            'num_queries_per_run':    nq_per_run,
        }

    print(f"\n✓ Aggregated {len(run_evals)} run(s) across k=1..{max(all_k)}\n")

    return {
        'metadata':  metadata,
        'k_results': aggregated_k,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_evaluation(evaluation: Dict[str, Any]):
    metadata  = evaluation['metadata']
    k_results = evaluation['k_results']

    print(f"\n{'='*70}")
    print(f"Random Baseline Evaluation Results")
    print(f"{'='*70}")
    print(f"Model:            {metadata['model']}")
    print(f"Token budget:     {metadata['token_budget']} tokens")
    print(f"Available for ICL:{metadata.get('available_for_icl', '?')} tokens")
    print(f"Pool size:        {metadata.get('pool_size', '?')} examples")
    print(f"Total queries:    {metadata['total_queries']}")
    print(f"Num runs:         {metadata.get('num_runs', 1)}")
    if 'seeds' in metadata:
        print(f"Seeds:            {metadata['seeds']}")
    print(f"{'='*70}\n")

    print(f"{'k':<5} | {'F1 %':>12} | {'Avg tokens':>12} | {'Queries (avg)':>14} | {'Runs w/ k':>9}")
    print(f"{'-'*65}")

    for k in sorted(k_results.keys(), key=int):
        r = k_results[k]
        f1_str  = f"{r['f1_percentage']:6.2f}±{r['f1_std_percentage']:4.2f}%"
        tok_str = f"{r['avg_cumulative_tokens']:8.1f} tok"
        nq_str  = f"{r['avg_num_queries']:8.1f}"
        print(f"{k:<5} | {f1_str:>12} | {tok_str:>12} | {nq_str:>14} | {r['num_runs_with_k']:>9}")

    # Summary: best k by F1
    best_k = max(k_results.keys(), key=lambda k: k_results[k]['avg_f1'])
    best   = k_results[best_k]
    print(f"\n✓ Best k by F1: k={best_k}  →  "
          f"F1={best['f1_percentage']:.2f}%  |  "
          f"avg tokens={best['avg_cumulative_tokens']:.1f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import re as _re

    parser = argparse.ArgumentParser(
        description="Evaluate random-baseline multi-turn ICL experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python evaluate_random_baseline_results.py \\
      --results-file random_baseline_multiturn/google_gemma-3-4b-it_random_baseline_budget_5000_multiturn_run_1_seed_42_results.json

  # Auto-discover and average 5 runs
  python evaluate_random_baseline_results.py \\
      --results-file random_baseline_multiturn/google_gemma-3-4b-it_random_baseline_budget_5000_multiturn_run_1_seed_42_results.json \\
      --num-runs 5

  # Save output
  python evaluate_random_baseline_results.py \\
      --results-file ...run_1_seed_42_results.json \\
      --num-runs 5 --output eval_random_gemma4b.json
        """
    )
    parser.add_argument("--results-file", type=str, required=True,
                        help="Path to any one run's result JSON (typically run 1). "
                             "Other runs are auto-discovered from the same directory.")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs to average (default: 1). "
                             "Automatically finds run_1 … run_N files.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save evaluation JSON to this file")

    args = parser.parse_args()

    base_path = Path(args.results_file)
    filename  = base_path.name

    if args.num_runs > 1:
        match = _re.search(r'(.+)_run_\d+_seed_\d+_results\.json', filename)
        if not match:
            print("ERROR: filename does not match expected pattern "
                  "*_run_N_seed_S_results.json")
            raise SystemExit(1)

        base_name    = match.group(1)
        result_files = []
        for run_idx in range(1, args.num_runs + 1):
            pattern = base_path.parent / f"{base_name}_run_{run_idx}_seed_*_results.json"
            matches = sorted(glob_module.glob(str(pattern)))
            if matches:
                result_files.append(matches[0])
            else:
                print(f"WARNING: no file found for run {run_idx} "
                      f"(pattern: {pattern})")

        if not result_files:
            print("ERROR: could not find any run files.")
            raise SystemExit(1)

        if len(result_files) != args.num_runs:
            print(f"WARNING: expected {args.num_runs} runs, found {len(result_files)}")

        print(f"\nFound {len(result_files)} run file(s):")
        for f in result_files:
            print(f"  {f}")
    else:
        result_files = [args.results_file]

    evaluation = evaluate_multiple_runs(result_files)
    print_evaluation(evaluation)

    if args.output:
        out = {
            'metadata':  evaluation['metadata'],
            'k_results': {str(k): v for k, v in evaluation['k_results'].items()},
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"✓ Evaluation saved to: {args.output}")

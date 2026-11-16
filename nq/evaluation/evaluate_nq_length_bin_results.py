#!/usr/bin/env python3
"""
Evaluate NQ Length Bin Experiment Results

Loads golden truth from nq_dev_golden_truth.jsonl (short_answers, long_answer,
yes_no_majority, is_unanswerable) and computes metrics against LLM responses.

Metrics per (bin, k):
  - Answerable F1  : token-level F1, max over all gold short answers + long answer
  - Answerable EM  : exact match, max over all gold answers
  - NO_ANSWER accuracy : for unanswerable questions
  - YES/NO accuracy    : for yes/no questions
  - Long-answer F1     : for questions with only a long answer (no short)

F1 is computed as max(F1(pred, g) for g in all_gold_answers), where
all_gold_answers = short_answers + [long_answer] (when available).
"""

import json
import string
import re
import argparse
import glob
import os
import statistics
from typing import Dict, List, Any
from pathlib import Path

GOLDEN_TRUTH_PATH = "/mnt/gpu-fastdata/anxo/sigir26-short-QA-ICL/nq-dataset-format/dataset-dev-example/nq_dev_golden_truth.jsonl"


# ---------------------------------------------------------------------------
# Text normalisation and metric helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles / punctuation / extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def extract_answer(llm_response: str) -> str:
    """Remove 'ANSWER:' prefix (case-insensitive) and strip."""
    return re.sub(r'^ANSWER:\s*', '', llm_response.strip(), flags=re.IGNORECASE).strip()


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Token-level F1 — max over all ground truths."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)
    return max_f1


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """1.0 if normalised prediction matches any ground truth exactly."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_norm == normalize_answer(gt):
            return 1.0
    return 0.0


def is_no_answer(text: str) -> bool:
    norm = normalize_answer(text)
    return norm in ('no_answer', 'noanswer', 'no answer', '')


def is_yes_no(text: str) -> bool:
    norm = normalize_answer(text)
    return norm in ('yes', 'no')


# ---------------------------------------------------------------------------
# Load golden truth
# ---------------------------------------------------------------------------

def load_golden_truth(path: str) -> Dict[int, Dict]:
    """Load golden truth keyed by example_id."""
    golden = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            golden[rec['example_id']] = rec
    return golden


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def evaluate_nq_results(results_file: str, golden_truth_path: str,
                        verbose: bool = True) -> Dict[str, Any]:
    """Evaluate a single NQ length-bin results file."""
    if verbose:
        print(f"\nEvaluating: {results_file}")

    golden = load_golden_truth(golden_truth_path)

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    metadata = results['metadata']
    experiments = results['experiments']

    if verbose:
        print(f"  Model: {metadata['model']}")
        print(f"  Token budget: {metadata['token_budget']} tokens")
        print(f"  Total queries: {metadata['total_queries']}")
        print(f"  Experiment configs present: {len(experiments)}")
        print(f"  Golden truth entries loaded: {len(golden)}")

    bin_k_metrics: Dict[int, Dict] = {}

    for exp in experiments:
        bin_id = exp['bin_id']
        k = exp['k']
        bin_range = exp['bin_range']

        if bin_id not in bin_k_metrics:
            bin_k_metrics[bin_id] = {'bin_range': bin_range, 'k_results': {}}

        metrics = {
            'answerable': {'f1_scores': [], 'em_scores': []},
            'unanswerable': {'correct': 0, 'total': 0, 'incorrect': 0},
            'yes_no': {'correct': 0, 'total': 0, 'incorrect': 0},
            'long_only': {'f1_scores': [], 'em_scores': []},
            'prompt_tokens': [],
            'example_token_lengths': [],
            'missing_gold': 0,
        }

        for resp in exp['responses']:
            eid = resp['example_id']
            llm_response = resp['llm_response']
            predicted = extract_answer(llm_response)

            # Token stats
            if 'conversation_tokens' in resp:
                metrics['prompt_tokens'].append(resp['conversation_tokens'])
            if 'example_token_lengths' in resp:
                metrics['example_token_lengths'].extend(resp['example_token_lengths'])

            gold = golden.get(eid)
            if gold is None:
                metrics['missing_gold'] += 1
                continue

            # --- Classify question type ---
            if gold['is_unanswerable']:
                metrics['unanswerable']['total'] += 1
                if is_no_answer(predicted):
                    metrics['unanswerable']['correct'] += 1
                else:
                    metrics['unanswerable']['incorrect'] += 1

            elif gold.get('yes_no_majority'):
                metrics['yes_no']['total'] += 1
                pred_norm = normalize_answer(predicted)
                gold_yn = gold['yes_no_majority'].lower()
                if pred_norm == gold_yn:
                    metrics['yes_no']['correct'] += 1
                else:
                    metrics['yes_no']['incorrect'] += 1

            elif gold['short_answers']:
                # Has short answers — build all gold candidates
                all_golds = list(gold['short_answers'])
                if gold.get('long_answer'):
                    all_golds.append(gold['long_answer'])

                if is_no_answer(predicted):
                    f1, em = 0.0, 0.0
                else:
                    f1 = compute_f1(predicted, all_golds)
                    em = compute_exact_match(predicted, all_golds)

                metrics['answerable']['f1_scores'].append(f1)
                metrics['answerable']['em_scores'].append(em)

            elif gold.get('long_answer'):
                # Long answer only (no short answers)
                all_golds = [gold['long_answer']]

                if is_no_answer(predicted):
                    f1, em = 0.0, 0.0
                else:
                    f1 = compute_f1(predicted, all_golds)
                    em = compute_exact_match(predicted, all_golds)

                metrics['long_only']['f1_scores'].append(f1)
                metrics['long_only']['em_scores'].append(em)

        bin_k_metrics[bin_id]['k_results'][k] = metrics

    # Aggregate
    evaluation: Dict[str, Any] = {'metadata': metadata, 'bins': {}}

    for bin_id in sorted(bin_k_metrics.keys()):
        bdata = bin_k_metrics[bin_id]
        bin_range = bdata['bin_range']
        evaluation['bins'][bin_id] = {'bin_range': bin_range, 'k_results': {}}

        for k in sorted(bdata['k_results'].keys()):
            m = bdata['k_results'][k]

            f1s = m['answerable']['f1_scores']
            ems = m['answerable']['em_scores']
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
            avg_em = sum(ems) / len(ems) if ems else 0.0

            lo_f1s = m['long_only']['f1_scores']
            lo_ems = m['long_only']['em_scores']
            avg_lo_f1 = sum(lo_f1s) / len(lo_f1s) if lo_f1s else 0.0
            avg_lo_em = sum(lo_ems) / len(lo_ems) if lo_ems else 0.0

            ut = m['unanswerable']['total']
            uc = m['unanswerable']['correct']
            no_ans_acc = (uc / ut * 100) if ut > 0 else 0.0

            ynt = m['yes_no']['total']
            ync = m['yes_no']['correct']
            yn_acc = (ync / ynt * 100) if ynt > 0 else 0.0

            pt = m['prompt_tokens']
            avg_pt = sum(pt) / len(pt) if pt else 0.0
            et = m['example_token_lengths']
            avg_et = sum(et) / len(et) if et else 0.0

            evaluation['bins'][bin_id]['k_results'][k] = {
                'k': k,
                'bin_range': bin_range,
                'answerable': {
                    'count': len(f1s),
                    'avg_f1': avg_f1,
                    'avg_em': avg_em,
                    'f1_percentage': avg_f1 * 100,
                    'em_percentage': avg_em * 100,
                },
                'long_only': {
                    'count': len(lo_f1s),
                    'avg_f1': avg_lo_f1,
                    'avg_em': avg_lo_em,
                    'f1_percentage': avg_lo_f1 * 100,
                    'em_percentage': avg_lo_em * 100,
                },
                'unanswerable': {
                    'count': ut,
                    'no_answer_accuracy': no_ans_acc,
                    'correct': uc,
                    'incorrect': m['unanswerable']['incorrect'],
                },
                'yes_no': {
                    'count': ynt,
                    'accuracy': yn_acc,
                    'correct': ync,
                    'incorrect': m['yes_no']['incorrect'],
                },
                'token_usage': {
                    'avg_prompt_tokens': avg_pt,
                    'avg_example_tokens': avg_et,
                },
                'missing_gold': m['missing_gold'],
            }

    return evaluation


# ---------------------------------------------------------------------------
# Multi-run averaged evaluation
# ---------------------------------------------------------------------------

def evaluate_multiple_runs(result_files: List[str],
                           golden_truth_path: str) -> Dict[str, Any]:
    """Average metrics across multiple runs."""
    print(f"\n{'='*80}")
    print(f"Evaluating {len(result_files)} runs for averaged results")
    print(f"{'='*80}\n")

    all_evals = []
    for i, rf in enumerate(result_files, 1):
        print(f"Run {i}/{len(result_files)}: {Path(rf).name}")
        ev = evaluate_nq_results(rf, golden_truth_path, verbose=False)
        all_evals.append(ev)
        print(f"  ✓ Evaluated")

    metadata = all_evals[0]['metadata'].copy()
    metadata['num_runs'] = len(result_files)
    metadata['seeds'] = [e['metadata'].get('seed', 'unknown') for e in all_evals]

    aggregated: Dict[str, Any] = {'metadata': metadata, 'bins': {}}

    first = all_evals[0]
    for bin_id in sorted(first['bins'].keys()):
        bin_range = first['bins'][bin_id]['bin_range']
        aggregated['bins'][bin_id] = {'bin_range': bin_range, 'k_results': {}}

        all_k = set()
        for ev in all_evals:
            if bin_id in ev['bins']:
                all_k.update(ev['bins'][bin_id]['k_results'].keys())

        for k in sorted(all_k, key=int):
            f1_runs, em_runs = [], []
            lo_f1_runs, lo_em_runs = [], []
            na_acc_runs, yn_acc_runs = [], []
            pt_runs, et_runs = [], []
            query_counts = []

            for ev in all_evals:
                if bin_id in ev['bins'] and k in ev['bins'][bin_id]['k_results']:
                    r = ev['bins'][bin_id]['k_results'][k]
                    f1_runs.append(r['answerable']['avg_f1'])
                    em_runs.append(r['answerable']['avg_em'])
                    lo_f1_runs.append(r['long_only']['avg_f1'])
                    lo_em_runs.append(r['long_only']['avg_em'])
                    na_acc_runs.append(r['unanswerable']['no_answer_accuracy'])
                    yn_acc_runs.append(r['yes_no']['accuracy'])
                    pt_runs.append(r['token_usage']['avg_prompt_tokens'])
                    et_runs.append(r['token_usage']['avg_example_tokens'])
                    query_counts.append(r['answerable']['count'])

            def _avg(lst): return sum(lst) / len(lst) if lst else 0.0
            def _std(lst): return statistics.stdev(lst) if len(lst) > 1 else 0.0

            aggregated['bins'][bin_id]['k_results'][k] = {
                'k': k,
                'bin_range': bin_range,
                'num_runs': len(f1_runs),
                'answerable': {
                    'count': _avg(query_counts),
                    'avg_f1': _avg(f1_runs), 'std_f1': _std(f1_runs),
                    'avg_em': _avg(em_runs), 'std_em': _std(em_runs),
                    'f1_percentage': _avg(f1_runs) * 100,
                    'f1_std_percentage': _std(f1_runs) * 100,
                    'em_percentage': _avg(em_runs) * 100,
                    'em_std_percentage': _std(em_runs) * 100,
                    'f1_per_run': f1_runs,
                    'em_per_run': em_runs,
                },
                'long_only': {
                    'count': 0,
                    'avg_f1': _avg(lo_f1_runs), 'std_f1': _std(lo_f1_runs),
                    'avg_em': _avg(lo_em_runs), 'std_em': _std(lo_em_runs),
                    'f1_percentage': _avg(lo_f1_runs) * 100,
                    'f1_std_percentage': _std(lo_f1_runs) * 100,
                    'em_percentage': _avg(lo_em_runs) * 100,
                    'em_std_percentage': _std(lo_em_runs) * 100,
                },
                'unanswerable': {
                    'count': 0,
                    'no_answer_accuracy': _avg(na_acc_runs),
                    'no_answer_accuracy_std': _std(na_acc_runs),
                    'correct': 0, 'incorrect': 0,
                },
                'yes_no': {
                    'count': 0,
                    'accuracy': _avg(yn_acc_runs),
                    'accuracy_std': _std(yn_acc_runs),
                    'correct': 0, 'incorrect': 0,
                },
                'token_usage': {
                    'avg_prompt_tokens': _avg(pt_runs),
                    'avg_example_tokens': _avg(et_runs),
                },
            }

    print(f"\n✓ Aggregated results from {len(result_files)} runs\n")
    return aggregated


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_evaluation(evaluation: Dict[str, Any], show_std: bool = True):
    meta = evaluation['metadata']

    print(f"\n{'='*100}")
    print(f"NQ Length Bin Experiment Evaluation Results")
    print(f"{'='*100}")
    print(f"Model: {meta['model']}")
    print(f"Token Budget: {meta['token_budget']} tokens")
    print(f"Instruction Overhead: {meta['instruction_overhead']} tokens")
    print(f"Available for Examples: {meta['available_for_examples']} tokens")
    print(f"Total Queries: {meta['total_queries']}")
    if 'num_runs' in meta:
        print(f"Number of Runs: {meta['num_runs']}")
        if 'seeds' in meta:
            print(f"Seeds: {', '.join(map(str, meta['seeds']))}")
    print(f"{'='*100}\n")

    for bin_id in sorted(evaluation['bins'].keys()):
        bdata = evaluation['bins'][bin_id]
        bin_range = bdata['bin_range']

        print(f"\n{'='*100}")
        print(f"BIN {bin_id}: Token Length Range [{bin_range[0]} - {bin_range[1]}]")
        print(f"{'='*100}")

        first_r = list(bdata['k_results'].values())[0]
        has_std = 'std_f1' in first_r['answerable']

        if has_std and show_std:
            header = (f"{'k':<4} | {'Avg F1':<15} | {'Avg EM':<15} | "
                      f"{'NO_ANS Acc':<15} | {'YES/NO Acc':<15} | "
                      f"{'LongOnly F1':<15} | {'Avg Prompt':<12} | {'Runs':<5}")
        else:
            header = (f"{'k':<4} | {'Avg F1':<10} | {'Avg EM':<10} | "
                      f"{'NO_ANS Acc':<12} | {'YES/NO Acc':<12} | "
                      f"{'LongOnly F1':<12} | {'Avg Prompt':<12} | "
                      f"{'Ans':<6} | {'Unans':<6} | {'Y/N':<5} | {'Long':<5}")
        print(f"\n{header}")
        print(f"{'-'*len(header)}")

        for k in sorted(bdata['k_results'].keys(), key=int):
            r = bdata['k_results'][k]
            f1_pct = r['answerable']['f1_percentage']
            em_pct = r['answerable']['em_percentage']
            na_acc = r['unanswerable']['no_answer_accuracy']
            yn_acc = r['yes_no']['accuracy']
            lo_f1 = r['long_only']['f1_percentage']
            avg_pt = r['token_usage']['avg_prompt_tokens']

            if has_std and show_std:
                f1_std = r['answerable'].get('f1_std_percentage', 0.0)
                em_std = r['answerable'].get('em_std_percentage', 0.0)
                na_std = r['unanswerable'].get('no_answer_accuracy_std', 0.0)
                yn_std = r['yes_no'].get('accuracy_std', 0.0)
                lo_std = r['long_only'].get('f1_std_percentage', 0.0)
                num_runs = r.get('num_runs', 1)
                print(f"{k:<4} | {f1_pct:>6.2f}±{f1_std:>4.2f}%  | "
                      f"{em_pct:>6.2f}±{em_std:>4.2f}%  | "
                      f"{na_acc:>6.2f}±{na_std:>4.2f}%  | "
                      f"{yn_acc:>6.2f}±{yn_std:>4.2f}%  | "
                      f"{lo_f1:>6.2f}±{lo_std:>4.2f}%  | "
                      f"{avg_pt:>8.1f} tok | {num_runs:<5}")
            else:
                ans_c = r['answerable']['count']
                una_c = r['unanswerable']['count']
                yn_c = r['yes_no']['count']
                lo_c = r['long_only']['count']
                print(f"{k:<4} | {f1_pct:>6.2f}%    | {em_pct:>6.2f}%    | "
                      f"{na_acc:>8.2f}%    | {yn_acc:>8.2f}%    | "
                      f"{lo_f1:>8.2f}%    | {avg_pt:>8.1f} tok | "
                      f"{ans_c:<6} | {una_c:<6} | {yn_c:<5} | {lo_c:<5}")

        best_k = max(bdata['k_results'].keys(),
                     key=lambda k: bdata['k_results'][k]['answerable']['avg_f1'])
        best_f1 = bdata['k_results'][best_k]['answerable']['f1_percentage']
        print(f"\n✓ Best k for Bin {bin_id}: k={best_k} (F1: {best_f1:.2f}%)")

    # Cross-bin summary
    print(f"\n\n{'='*100}")
    print(f"Cross-Bin Comparison (best k per bin)")
    print(f"{'='*100}\n")

    first_bin = list(evaluation['bins'].values())[0]
    first_r = list(first_bin['k_results'].values())[0]
    has_std = 'std_f1' in first_r['answerable']

    print(f"{'Bin':<6} | {'k':<4} | {'Token Range':<16} | {'Avg F1':<15} | {'Avg EM':<15} | {'NO_ANS':<12} | {'YES/NO':<12} | {'LongOnly F1':<12}")
    print(f"{'-'*120}")

    for bin_id in sorted(evaluation['bins'].keys(), key=int):
        bdata = evaluation['bins'][bin_id]
        bin_range = bdata['bin_range']

        for k in sorted(bdata['k_results'].keys(), key=int):
            r = bdata['k_results'][k]
            f1_pct = r['answerable']['f1_percentage']
            em_pct = r['answerable']['em_percentage']
            na_acc = r['unanswerable']['no_answer_accuracy']
            yn_acc = r['yes_no']['accuracy']
            lo_f1 = r['long_only']['f1_percentage']

            if has_std and show_std:
                f1_std = r['answerable'].get('f1_std_percentage', 0.0)
                em_std = r['answerable'].get('em_std_percentage', 0.0)
                print(f"{bin_id:<6} | {k:<4} | [{bin_range[0]}-{bin_range[1]}]"
                      f"{'':>{14-len(str(bin_range[0]))-len(str(bin_range[1]))}} | "
                      f"{f1_pct:>6.2f}±{f1_std:>4.2f}%  | "
                      f"{em_pct:>6.2f}±{em_std:>4.2f}%  | "
                      f"{na_acc:>8.2f}%    | {yn_acc:>8.2f}%    | {lo_f1:>8.2f}%")
            else:
                print(f"{bin_id:<6} | {k:<4} | [{bin_range[0]}-{bin_range[1]}]"
                      f"{'':>{14-len(str(bin_range[0]))-len(str(bin_range[1]))}} | "
                      f"{f1_pct:>6.2f}%         | {em_pct:>6.2f}%         | "
                      f"{na_acc:>8.2f}%    | {yn_acc:>8.2f}%    | {lo_f1:>8.2f}%")

    # Overall best
    print(f"\n\n{'='*80}")
    print(f"Summary: Key Findings")
    print(f"{'='*80}\n")

    best_bin, best_k_overall, best_f1_overall = None, None, 0.0
    for bin_id in evaluation['bins']:
        for k in evaluation['bins'][bin_id]['k_results']:
            f1 = evaluation['bins'][bin_id]['k_results'][k]['answerable']['avg_f1']
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_bin = bin_id
                best_k_overall = k

    if best_bin is not None:
        br = evaluation['bins'][best_bin]['k_results'][best_k_overall]
        bin_range = br['bin_range']
        print(f"✓ Best Overall Configuration:")
        print(f"  - Bin {best_bin} (token length [{bin_range[0]}-{bin_range[1]}])")
        print(f"  - k = {best_k_overall} examples")
        print(f"  - F1: {br['answerable']['f1_percentage']:.2f}%")
        print(f"  - EM: {br['answerable']['em_percentage']:.2f}%")
        print(f"  - NO_ANSWER accuracy: {br['unanswerable']['no_answer_accuracy']:.2f}%")
        print(f"  - YES/NO accuracy: {br['yes_no']['accuracy']:.2f}%")
        print(f"  - Long-only F1: {br['long_only']['f1_percentage']:.2f}%")

    print(f"\n✓ Best k per Bin:")
    for bin_id in sorted(evaluation['bins'].keys()):
        bd = evaluation['bins'][bin_id]
        bk = max(bd['k_results'].keys(),
                 key=lambda k: bd['k_results'][k]['answerable']['avg_f1'])
        br = bd['k_results'][bk]
        bin_range = br['bin_range']
        print(f"  - Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: k={bk} "
              f"(F1: {br['answerable']['f1_percentage']:.2f}%, "
              f"EM: {br['answerable']['em_percentage']:.2f}%)")

    print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NQ length bin experiment results against golden truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single run
  python evaluate_nq_length_bin_results.py \\
      --results-file results/google_..._run_1_seed_42_results.json

  # Average over 5 runs
  python evaluate_nq_length_bin_results.py \\
      --results-file results/google_..._run_1_seed_42_results.json \\
      --num-runs 5

  # Evaluate all files in a directory
  python evaluate_nq_length_bin_results.py \\
      --results-dir results/ \\
      --pattern "*_nq_length_bin_*_results.json"

  # Custom golden truth path
  python evaluate_nq_length_bin_results.py \\
      --results-file results.json \\
      --golden-truth /path/to/nq_dev_golden_truth.jsonl
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-file", type=str,
                       help="Path to a results JSON file (single run or first of multiple)")
    group.add_argument("--results-dir", type=str,
                       help="Directory containing results files")

    parser.add_argument("--golden-truth", type=str, default=GOLDEN_TRUTH_PATH,
                        help=f"Path to golden truth JSONL (default: {GOLDEN_TRUTH_PATH})")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs to average (default: 1)")
    parser.add_argument("--pattern", type=str,
                        default="*_nq_length_bin_*_results.json",
                        help="Glob pattern for --results-dir mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Save evaluation results to JSON file")

    args = parser.parse_args()

    # Collect results files
    if args.results_file:
        if args.num_runs > 1:
            base_path = Path(args.results_file)
            filename = base_path.name
            match = re.search(r'(.+)_run_\d+_seed_\d+_results\.json', filename)
            if not match:
                print(f"\n✗ Filename does not match *_run_X_seed_Y_results.json pattern")
                print(f"   Got: {filename}")
                return

            base_name = match.group(1)
            result_files = []
            for run_idx in range(1, args.num_runs + 1):
                pattern = base_path.parent / f"{base_name}_run_{run_idx}_seed_*_results.json"
                matches = glob.glob(str(pattern))
                if matches:
                    result_files.append(matches[0])
                else:
                    print(f"Warning: No file for run {run_idx} matching {pattern}")

            if len(result_files) != args.num_runs:
                print(f"\n✗ Expected {args.num_runs} run files, found {len(result_files)}")
                return

            print(f"\n{'='*80}")
            print(f"Found {len(result_files)} run files:")
            print(f"{'='*80}")
            for f in result_files:
                print(f"  - {Path(f).name}")
            print(f"{'='*80}\n")
        else:
            result_files = [args.results_file]
    else:
        pattern = os.path.join(args.results_dir, args.pattern)
        result_files = sorted(glob.glob(pattern))
        if not result_files:
            print(f"\n✗ No files matching: {pattern}")
            return
        print(f"\n{'='*80}")
        print(f"Found {len(result_files)} result files")
        print(f"{'='*80}")
        for f in result_files:
            print(f"  - {os.path.basename(f)}")
        print(f"{'='*80}\n")

    # Evaluate
    if args.num_runs > 1 and args.results_file:
        evaluation = evaluate_multiple_runs(result_files, args.golden_truth)
        print_evaluation(evaluation, show_std=True)
        all_evaluations = {f"averaged_{args.num_runs}_runs": evaluation}
    else:
        all_evaluations = {}
        for rf in result_files:
            fname = os.path.basename(rf)
            print(f"\n{'='*80}")
            print(f"Evaluating: {fname}")
            print(f"{'='*80}")
            ev = evaluate_nq_results(rf, args.golden_truth, verbose=True)
            print_evaluation(ev, show_std=False)
            all_evaluations[fname] = ev

    # Save
    if args.output:
        output_data = {
            'evaluations': all_evaluations,
            'summary': {
                'total_files': len(all_evaluations),
                'files': list(all_evaluations.keys()),
            },
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Evaluation saved to: {args.output}\n")


if __name__ == "__main__":
    main()

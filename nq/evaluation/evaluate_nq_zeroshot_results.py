#!/usr/bin/env python3
"""
Evaluate NQ Zero-Shot Experiment Results

Loads golden truth from nq_dev_golden_truth.jsonl (short_answers, long_answer,
yes_no_majority, is_unanswerable) and computes metrics against LLM responses
from zero-shot experiments (no ICL examples, flat response list).

Metrics:
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_nq_zeroshot(results_file: str, golden_truth_path: str,
                         verbose: bool = True) -> Dict[str, Any]:
    """Evaluate a single NQ zero-shot results file."""
    if verbose:
        print(f"\nEvaluating: {results_file}")

    golden = load_golden_truth(golden_truth_path)

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    metadata = results['metadata']
    responses = results['responses']

    if verbose:
        print(f"  Model: {metadata['model']}")
        print(f"  Approach: {metadata['approach']}")
        print(f"  Total queries: {metadata['total_queries']}")
        print(f"  Golden truth entries loaded: {len(golden)}")

    metrics = {
        'answerable': {'f1_scores': [], 'em_scores': []},
        'unanswerable': {'correct': 0, 'total': 0, 'incorrect': 0},
        'yes_no': {'correct': 0, 'total': 0, 'incorrect': 0},
        'long_only': {'f1_scores': [], 'em_scores': []},
        'missing_gold': 0,
    }

    for resp in responses:
        eid = resp['example_id']
        llm_response = resp['llm_response']
        predicted = extract_answer(llm_response)

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
            all_golds = [gold['long_answer']]

            if is_no_answer(predicted):
                f1, em = 0.0, 0.0
            else:
                f1 = compute_f1(predicted, all_golds)
                em = compute_exact_match(predicted, all_golds)

            metrics['long_only']['f1_scores'].append(f1)
            metrics['long_only']['em_scores'].append(em)

    # Aggregate
    f1s = metrics['answerable']['f1_scores']
    ems = metrics['answerable']['em_scores']
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_em = sum(ems) / len(ems) if ems else 0.0

    lo_f1s = metrics['long_only']['f1_scores']
    lo_ems = metrics['long_only']['em_scores']
    avg_lo_f1 = sum(lo_f1s) / len(lo_f1s) if lo_f1s else 0.0
    avg_lo_em = sum(lo_ems) / len(lo_ems) if lo_ems else 0.0

    ut = metrics['unanswerable']['total']
    uc = metrics['unanswerable']['correct']
    no_ans_acc = (uc / ut * 100) if ut > 0 else 0.0

    ynt = metrics['yes_no']['total']
    ync = metrics['yes_no']['correct']
    yn_acc = (ync / ynt * 100) if ynt > 0 else 0.0

    evaluation = {
        'metadata': metadata,
        'results': {
            'total_queries': len(responses),
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
                'incorrect': metrics['unanswerable']['incorrect'],
            },
            'yes_no': {
                'count': ynt,
                'accuracy': yn_acc,
                'correct': ync,
                'incorrect': metrics['yes_no']['incorrect'],
            },
            'missing_gold': metrics['missing_gold'],
        },
    }

    return evaluation


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_evaluation(evaluation: Dict[str, Any]):
    meta = evaluation['metadata']
    r = evaluation['results']

    print(f"\n{'='*80}")
    print(f"NQ Zero-Shot Experiment Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {meta['model']}")
    print(f"Approach: {meta['approach']}")
    print(f"Total Queries: {r['total_queries']}")
    print(f"{'='*80}\n")

    header = (f"{'Metric':<20} | {'Value':<12} | {'Count':<8}")
    print(header)
    print(f"{'-'*len(header)}")

    print(f"{'Answerable F1':<20} | {r['answerable']['f1_percentage']:>8.2f}%    | {r['answerable']['count']:<8}")
    print(f"{'Answerable EM':<20} | {r['answerable']['em_percentage']:>8.2f}%    | {r['answerable']['count']:<8}")
    print(f"{'NO_ANSWER Acc':<20} | {r['unanswerable']['no_answer_accuracy']:>8.2f}%    | {r['unanswerable']['count']:<8}")
    print(f"{'YES/NO Acc':<20} | {r['yes_no']['accuracy']:>8.2f}%    | {r['yes_no']['count']:<8}")
    print(f"{'Long-only F1':<20} | {r['long_only']['f1_percentage']:>8.2f}%    | {r['long_only']['count']:<8}")

    if r['missing_gold'] > 0:
        print(f"\n⚠ Missing golden truth for {r['missing_gold']} queries")

    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}\n")
    print(f"✓ Answerable F1:      {r['answerable']['f1_percentage']:.2f}%  ({r['answerable']['count']} questions)")
    print(f"✓ Answerable EM:      {r['answerable']['em_percentage']:.2f}%")
    print(f"✓ NO_ANSWER accuracy: {r['unanswerable']['no_answer_accuracy']:.2f}%  ({r['unanswerable']['count']} questions)")
    print(f"✓ YES/NO accuracy:    {r['yes_no']['accuracy']:.2f}%  ({r['yes_no']['count']} questions)")
    print(f"✓ Long-only F1:       {r['long_only']['f1_percentage']:.2f}%  ({r['long_only']['count']} questions)")
    print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NQ zero-shot experiment results against golden truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single file
  python evaluate_nq_zeroshot_results.py \\
      --results-file results/google_..._nq_zeroshot_results.json

  # Evaluate all files in a directory
  python evaluate_nq_zeroshot_results.py \\
      --results-dir NQ_code/get_token_length_size/zeroshot/

  # Custom golden truth path
  python evaluate_nq_zeroshot_results.py \\
      --results-file results.json \\
      --golden-truth /path/to/nq_dev_golden_truth.jsonl
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-file", type=str,
                       help="Path to a single zero-shot results JSON file")
    group.add_argument("--results-dir", type=str,
                       help="Directory containing results files")

    parser.add_argument("--golden-truth", type=str, default=GOLDEN_TRUTH_PATH,
                        help=f"Path to golden truth JSONL (default: {GOLDEN_TRUTH_PATH})")
    parser.add_argument("--pattern", type=str,
                        default="*_nq_zeroshot_results.json",
                        help="Glob pattern for --results-dir mode")
    parser.add_argument("--output", type=str, default=None,
                        help="Save evaluation results to JSON file")

    args = parser.parse_args()

    # Collect results files
    if args.results_file:
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
    all_evaluations = {}
    for rf in result_files:
        fname = os.path.basename(rf)
        print(f"\n{'='*80}")
        print(f"Evaluating: {fname}")
        print(f"{'='*80}")
        ev = evaluate_nq_zeroshot(rf, args.golden_truth, verbose=True)
        print_evaluation(ev)
        all_evaluations[fname] = ev

    # Cross-model comparison if multiple files
    if len(all_evaluations) > 1:
        print(f"\n{'='*80}")
        print(f"Cross-Model Comparison (Zero-Shot)")
        print(f"{'='*80}\n")

        header = (f"{'Model':<45} | {'F1':<10} | {'EM':<10} | "
                  f"{'NO_ANS':<10} | {'YES/NO':<10} | {'LongF1':<10}")
        print(header)
        print(f"{'-'*len(header)}")

        for fname, ev in all_evaluations.items():
            model = ev['metadata']['model']
            r = ev['results']
            print(f"{model:<45} | {r['answerable']['f1_percentage']:>6.2f}%  | "
                  f"{r['answerable']['em_percentage']:>6.2f}%  | "
                  f"{r['unanswerable']['no_answer_accuracy']:>6.2f}%  | "
                  f"{r['yes_no']['accuracy']:>6.2f}%  | "
                  f"{r['long_only']['f1_percentage']:>6.2f}%")
        print(f"\n{'='*80}\n")

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

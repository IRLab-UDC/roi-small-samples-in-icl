#!/usr/bin/env python3
"""
Evaluate Zero-Shot Experiment Results for Beat the AI

Since zero-shot has no bins and no k, the output is a single set of metrics
per results file:
  - Average F1 (answerable)
  - Average EM (answerable)
  - NO_ANSWER accuracy (unanswerable)
  - Total / answerable / unanswerable counts

Usage:
  # Single file
  python evaluate_zero_shot_results.py \
      --results-file zero_shot_experiments/google_gemma-3-4b-it_zero_shot_results.json

  # Multiple files (evaluates each independently)
  python evaluate_zero_shot_results.py \
      --results-dir zero_shot_experiments/

  # Save JSON output
  python evaluate_zero_shot_results.py \
      --results-file ... --output eval_zeroshot.json
"""

import argparse
import glob as glob_module
import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List, Any


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


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_norm == normalize_answer(gt):
            return 1.0
    return 0.0


def is_no_answer(text: str) -> bool:
    return normalize_answer(text) in {'no_answer', 'noanswer', 'no answer', ''}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_zero_shot(results_file: str) -> Dict[str, Any]:
    """Evaluate a single zero-shot results file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata  = data['metadata']
    responses = data['responses']

    f1_scores  = []
    em_scores  = []
    ans_count  = 0
    unans_count = 0
    no_ans_correct = 0
    no_ans_incorrect = 0

    for resp in responses:
        is_impossible = resp['is_impossible']
        answers_text  = resp['answers_text']
        predicted     = extract_answer(resp['llm_response'])

        if is_impossible:
            unans_count += 1
            if is_no_answer(predicted):
                no_ans_correct += 1
            else:
                no_ans_incorrect += 1
        else:
            ans_count += 1
            if is_no_answer(predicted):
                f1_scores.append(0.0)
                em_scores.append(0.0)
            else:
                f1_scores.append(compute_f1(predicted, answers_text))
                em_scores.append(compute_exact_match(predicted, answers_text))

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    no_ans_acc = (no_ans_correct / unans_count * 100) if unans_count > 0 else 0.0

    return {
        'metadata': metadata,
        'results': {
            'total_queries': len(responses),
            'answerable_count': ans_count,
            'unanswerable_count': unans_count,
            'avg_f1': avg_f1,
            'f1_percentage': avg_f1 * 100,
            'avg_em': avg_em,
            'em_percentage': avg_em * 100,
            'no_answer_accuracy': no_ans_acc,
            'no_answer_correct': no_ans_correct,
            'no_answer_incorrect': no_ans_incorrect,
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_evaluation(evaluation: Dict[str, Any], filename: str = ""):
    meta = evaluation['metadata']
    res  = evaluation['results']

    print(f"\n{'='*60}")
    print(f"Zero-Shot Evaluation Results")
    if filename:
        print(f"File: {filename}")
    print(f"{'='*60}")
    print(f"Model:          {meta['model']}")
    print(f"Approach:       {meta.get('approach', 'zero-shot')}")
    print(f"Total queries:  {res['total_queries']}")
    print(f"  Answerable:   {res['answerable_count']}")
    print(f"  Unanswerable: {res['unanswerable_count']}")
    print(f"{'='*60}")
    print(f"  Avg F1:              {res['f1_percentage']:6.2f}%")
    print(f"  Avg EM:              {res['em_percentage']:6.2f}%")
    if res['unanswerable_count'] > 0:
        print(f"  NO_ANSWER accuracy:  {res['no_answer_accuracy']:6.2f}%  "
              f"({res['no_answer_correct']}/{res['unanswerable_count']})")
    else:
        print(f"  NO_ANSWER accuracy:  N/A (no unanswerable questions)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot experiment results for Beat the AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python evaluate_zero_shot_results.py \\
      --results-file zero_shot_experiments/google_gemma-3-4b-it_zero_shot_results.json

  # All files in a directory
  python evaluate_zero_shot_results.py \\
      --results-dir zero_shot_experiments/

  # Save output
  python evaluate_zero_shot_results.py \\
      --results-file ... --output eval_zeroshot.json
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-file", type=str,
                       help="Path to a zero-shot results JSON file")
    group.add_argument("--results-dir", type=str,
                       help="Directory containing zero-shot results files")

    parser.add_argument("--pattern", type=str,
                        default="*_zero_shot_results.json",
                        help="Glob pattern for --results-dir mode (default: *_zero_shot_results.json)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save evaluation JSON to this file")

    args = parser.parse_args()

    # Collect results files
    if args.results_file:
        result_files = [args.results_file]
    else:
        pattern = os.path.join(args.results_dir, args.pattern)
        result_files = sorted(glob_module.glob(pattern))
        if not result_files:
            print(f"\n✗ No files matching: {pattern}")
            return

    all_evaluations = {}
    for rf in result_files:
        fname = os.path.basename(rf)
        ev = evaluate_zero_shot(rf)
        print_evaluation(ev, filename=fname)
        all_evaluations[fname] = ev

    # Comparison table if multiple files
    if len(all_evaluations) > 1:
        print(f"\n{'='*80}")
        print(f"Comparison Across Models")
        print(f"{'='*80}\n")
        print(f"{'Model':<40} | {'F1 %':>8} | {'EM %':>8} | {'NO_ANS %':>10} | {'Queries':>8}")
        print(f"{'-'*85}")
        for fname, ev in all_evaluations.items():
            model = ev['metadata']['model']
            res = ev['results']
            na_str = f"{res['no_answer_accuracy']:6.2f}%" if res['unanswerable_count'] > 0 else "N/A"
            print(f"{model:<40} | {res['f1_percentage']:>6.2f}% | "
                  f"{res['em_percentage']:>6.2f}% | {na_str:>10} | {res['total_queries']:>8}")
        print()

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

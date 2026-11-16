#!/usr/bin/env python3
"""
Evaluate Length Bin Experiment Results

This script evaluates the results from length-based bin experiments to analyze
the impact of example token length on ICL performance.

For each bin and k value, it computes:
1. Average F1 score (answerable questions)
2. Average Exact Match (EM) score (answerable questions)
3. NO_ANSWER accuracy (unanswerable questions)
4. Overall performance metrics

This allows analyzing:
- How does example length affect performance?
- What is the optimal k for each bin?
- Does using more short examples beat fewer long examples?
"""

import json
import string
import re
import argparse
import glob
import os
import statistics
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for evaluation.
    
    Lower case, remove punctuation, articles, and extra whitespace.
    Based on official Beat the AI/QA evaluation script.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_answer(llm_response: str) -> str:
    """
    Extract answer from LLM response.
    
    Removes "ANSWER:" prefix if present and strips whitespace.
    
    Args:
        llm_response: Raw LLM response
        
    Returns:
        Extracted answer text
    """
    # Remove "ANSWER:" prefix (case insensitive)
    answer = re.sub(r'^ANSWER:\s*', '', llm_response.strip(), flags=re.IGNORECASE)
    return answer.strip()


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute F1 score between prediction and list of ground truth answers.
    
    Returns maximum F1 score across all ground truths.
    Based on official Beat the AI/QA evaluation script.
    
    Args:
        prediction: Predicted answer text
        ground_truths: List of acceptable answer texts
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    # Normalize
    prediction_tokens = normalize_answer(prediction).split()
    
    max_f1 = 0.0
    for ground_truth in ground_truths:
        ground_truth_tokens = normalize_answer(ground_truth).split()
        
        # Compute token overlap
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = len(common)
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
            recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
            f1 = (2 * precision * recall) / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute exact match score.
    
    Returns 1.0 if prediction matches any ground truth exactly (after normalization).
    
    Args:
        prediction: Predicted answer text
        ground_truths: List of acceptable answer texts
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    prediction_normalized = normalize_answer(prediction)
    for ground_truth in ground_truths:
        if prediction_normalized == normalize_answer(ground_truth):
            return 1.0
    return 0.0


def is_no_answer(text: str) -> bool:
    """
    Check if response indicates no answer.
    
    Args:
        text: Response text
        
    Returns:
        True if response is NO_ANSWER
    """
    normalized = normalize_answer(text)
    return normalized in ['no_answer', 'noanswer', 'no answer', '']


def evaluate_length_bin_results(results_file: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate length bin experiment results from a single run.
    
    Args:
        results_file: Path to results JSON file
        verbose: Print progress messages
        
    Returns:
        Evaluation metrics dictionary organized by bin_id and k
    """
    if verbose:
        print(f"\nEvaluating length bin results: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    metadata = results['metadata']
    experiments = results['experiments']
    
    if verbose:
        print(f"✓ Model: {metadata['model']}")
        print(f"✓ Token budget: {metadata['token_budget']} tokens")
        print(f"✓ Instruction overhead: {metadata['instruction_overhead']} tokens")
        print(f"✓ Available for examples: {metadata['available_for_examples']} tokens")
        print(f"✓ Total queries: {metadata['total_queries']}")
    
    # Initialize metrics structure
    bin_k_metrics = {}
    
    # Process each experiment (bin_id + k combination)
    for exp in experiments:
        bin_id = exp['bin_id']
        k = exp['k']
        bin_range = exp['bin_range']
        
        if bin_id not in bin_k_metrics:
            bin_k_metrics[bin_id] = {
                'bin_range': bin_range,
                'k_results': {}
            }
        
        # Initialize metrics for this k
        bin_k_metrics[bin_id]['k_results'][k] = {
            'answerable': {'f1_scores': [], 'em_scores': []},
            'unanswerable': {'correct_no_answer': 0, 'total': 0, 'incorrect_answer': 0},
            'prompt_tokens': [],
            'example_token_lengths': []
        }
        
        metrics = bin_k_metrics[bin_id]['k_results'][k]
        
        # Process each response
        for response in exp['responses']:
            is_impossible = response['is_impossible']
            answers_text = response['answers_text']
            llm_response = response['llm_response']
            
            # Track token usage (support both single-turn and multi-turn formats)
            if 'prompt_tokens' in response:
                metrics['prompt_tokens'].append(response['prompt_tokens'])
            elif 'conversation_tokens' in response:
                # Multi-turn format uses 'conversation_tokens' instead of 'prompt_tokens'
                metrics['prompt_tokens'].append(response['conversation_tokens'])
            
            if 'example_token_lengths' in response:
                metrics['example_token_lengths'].extend(response['example_token_lengths'])
            
            # Extract predicted answer
            predicted_answer = extract_answer(llm_response)
            
            if is_impossible:
                # Unanswerable question
                metrics['unanswerable']['total'] += 1
                
                if is_no_answer(predicted_answer):
                    metrics['unanswerable']['correct_no_answer'] += 1
                else:
                    metrics['unanswerable']['incorrect_answer'] += 1
            else:
                # Answerable question
                if is_no_answer(predicted_answer):
                    # False negative
                    f1_score = 0.0
                    em_score = 0.0
                else:
                    f1_score = compute_f1(predicted_answer, answers_text)
                    em_score = compute_exact_match(predicted_answer, answers_text)
                
                metrics['answerable']['f1_scores'].append(f1_score)
                metrics['answerable']['em_scores'].append(em_score)
    
    # Compute aggregated metrics
    evaluation = {
        'metadata': metadata,
        'bins': {}
    }
    
    for bin_id in sorted(bin_k_metrics.keys()):
        bin_data = bin_k_metrics[bin_id]
        bin_range = bin_data['bin_range']
        
        evaluation['bins'][bin_id] = {
            'bin_range': bin_range,
            'k_results': {}
        }
        
        for k in sorted(bin_data['k_results'].keys()):
            metrics = bin_data['k_results'][k]
            
            # Answerable metrics
            f1_scores = metrics['answerable']['f1_scores']
            em_scores = metrics['answerable']['em_scores']
            
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
            
            # Unanswerable metrics
            unans_total = metrics['unanswerable']['total']
            unans_correct = metrics['unanswerable']['correct_no_answer']
            unans_incorrect = metrics['unanswerable']['incorrect_answer']
            
            no_answer_accuracy = (unans_correct / unans_total * 100) if unans_total > 0 else 0.0
            
            # Token usage
            prompt_tokens = metrics['prompt_tokens']
            avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0
            
            example_tokens = metrics['example_token_lengths']
            avg_example_tokens = sum(example_tokens) / len(example_tokens) if example_tokens else 0.0
            
            evaluation['bins'][bin_id]['k_results'][k] = {
                'k': k,
                'bin_range': bin_range,
                'answerable': {
                    'count': len(f1_scores),
                    'avg_f1': avg_f1,
                    'avg_em': avg_em,
                    'f1_percentage': avg_f1 * 100,
                    'em_percentage': avg_em * 100
                },
                'unanswerable': {
                    'count': unans_total,
                    'no_answer_accuracy': no_answer_accuracy,
                    'correct_no_answer': unans_correct,
                    'incorrect_answer': unans_incorrect
                },
                'token_usage': {
                    'avg_prompt_tokens': avg_prompt_tokens,
                    'avg_example_tokens': avg_example_tokens
                }
            }
    
    return evaluation


def evaluate_multiple_runs(result_files: List[str]) -> Dict[str, Any]:
    """
    Evaluate multiple runs and compute averaged metrics with standard deviations.
    
    Args:
        result_files: List of paths to results JSON files from different runs
        
    Returns:
        Averaged evaluation metrics with standard deviations
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {len(result_files)} runs for averaged results")
    print(f"{'='*80}\n")
    
    # Evaluate each run
    all_evaluations = []
    for i, results_file in enumerate(result_files, 1):
        print(f"Run {i}/{len(result_files)}: {Path(results_file).name}")
        evaluation = evaluate_length_bin_results(results_file, verbose=False)
        all_evaluations.append(evaluation)
        print(f"  ✓ Evaluated")
    
    # Get metadata from first run (should be same across all runs except seed)
    metadata = all_evaluations[0]['metadata'].copy()
    metadata['num_runs'] = len(result_files)
    metadata['seeds'] = [eval_data['metadata'].get('seed', 'unknown') for eval_data in all_evaluations]
    
    # Aggregate results across runs
    aggregated = {
        'metadata': metadata,
        'bins': {}
    }
    
    # Get bin structure from first evaluation
    first_eval = all_evaluations[0]
    
    for bin_id in sorted(first_eval['bins'].keys()):
        bin_data = first_eval['bins'][bin_id]
        bin_range = bin_data['bin_range']
        
        aggregated['bins'][bin_id] = {
            'bin_range': bin_range,
            'k_results': {}
        }
        
        # Get all k values for this bin
        all_k_values = set()
        for eval_data in all_evaluations:
            if bin_id in eval_data['bins']:
                all_k_values.update(eval_data['bins'][bin_id]['k_results'].keys())
        
        # Aggregate each k
        for k in sorted(all_k_values, key=int):
            # Collect metrics from all runs for this bin+k
            f1_scores = []
            em_scores = []
            prompt_tokens = []
            example_tokens = []
            
            for eval_data in all_evaluations:
                if bin_id in eval_data['bins'] and k in eval_data['bins'][bin_id]['k_results']:
                    result = eval_data['bins'][bin_id]['k_results'][k]
                    f1_scores.append(result['answerable']['avg_f1'])
                    em_scores.append(result['answerable']['avg_em'])
                    prompt_tokens.append(result['token_usage']['avg_prompt_tokens'])
                    example_tokens.append(result['token_usage']['avg_example_tokens'])
            
            # Compute statistics
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            std_f1 = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
            
            avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
            std_em = statistics.stdev(em_scores) if len(em_scores) > 1 else 0.0
            
            avg_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0
            avg_example_tokens = sum(example_tokens) / len(example_tokens) if example_tokens else 0.0
            
            # Get count from first run (should be same across all runs)
            first_result = None
            for eval_data in all_evaluations:
                if bin_id in eval_data['bins'] and k in eval_data['bins'][bin_id]['k_results']:
                    first_result = eval_data['bins'][bin_id]['k_results'][k]
                    break
            
            count = first_result['answerable']['count'] if first_result else 0
            
            aggregated['bins'][bin_id]['k_results'][k] = {
                'k': k,
                'bin_range': bin_range,
                'num_runs': len(f1_scores),
                'answerable': {
                    'count': count,
                    'avg_f1': avg_f1,
                    'std_f1': std_f1,
                    'avg_em': avg_em,
                    'std_em': std_em,
                    'f1_percentage': avg_f1 * 100,
                    'f1_std_percentage': std_f1 * 100,
                    'em_percentage': avg_em * 100,
                    'em_std_percentage': std_em * 100,
                    # Also store individual run values for reference
                    'f1_per_run': f1_scores,
                    'em_per_run': em_scores
                },
                'unanswerable': {
                    'count': 0,
                    'no_answer_accuracy': 0.0,
                    'correct_no_answer': 0,
                    'incorrect_answer': 0
                },
                'token_usage': {
                    'avg_prompt_tokens': avg_prompt_tokens,
                    'avg_example_tokens': avg_example_tokens
                }
            }
    
    print(f"\n✓ Aggregated results from {len(result_files)} runs\n")
    return aggregated


def print_evaluation(evaluation: Dict[str, Any], show_std: bool = True):
    """
    Pretty print evaluation results with focus on bin and k analysis.
    
    Args:
        evaluation: Evaluation metrics dictionary
        show_std: Show standard deviations (for multi-run evaluations)
    """
    metadata = evaluation['metadata']
    
    print(f"\n{'='*80}")
    print(f"Length Bin Experiment Evaluation Results")
    print(f"{'='*80}")
    print(f"Model: {metadata['model']}")
    print(f"Token Budget: {metadata['token_budget']} tokens")
    print(f"Instruction Overhead: {metadata['instruction_overhead']} tokens")
    print(f"Available for Examples: {metadata['available_for_examples']} tokens")
    print(f"Total Queries: {metadata['total_queries']}")
    
    # Show run information if available
    if 'num_runs' in metadata:
        print(f"Number of Runs: {metadata['num_runs']}")
        if 'seeds' in metadata:
            print(f"Seeds: {', '.join(map(str, metadata['seeds']))}")
    
    print(f"{'='*80}\n")
    
    # Print results for each bin
    for bin_id in sorted(evaluation['bins'].keys()):
        bin_data = evaluation['bins'][bin_id]
        bin_range = bin_data['bin_range']
        
        print(f"\n{'='*80}")
        print(f"BIN {bin_id}: Token Length Range [{bin_range[0]} - {bin_range[1]}]")
        print(f"{'='*80}")
        
        # Check if this is multi-run evaluation
        first_result = list(bin_data['k_results'].values())[0]
        has_std = 'std_f1' in first_result['answerable']
        
        # Header
        if has_std and show_std:
            print(f"\n{'k':<4} | {'Avg F1':<15} | {'Avg EM':<15} | {'Avg Prompt':<12} | {'Avg Example':<13} | {'Runs':<6}")
            print(f"{'-'*90}")
        else:
            print(f"\n{'k':<4} | {'Avg F1':<10} | {'Avg EM':<10} | {'NO_ANS Acc':<12} | {'Avg Prompt':<12} | {'Avg Example':<13} | {'Ans':<6} | {'Unans':<6}")
            print(f"{'-'*100}")
        
        # Print each k value
        for k in sorted(bin_data['k_results'].keys(), key=int):
            result = bin_data['k_results'][k]
            
            f1_pct = result['answerable']['f1_percentage']
            em_pct = result['answerable']['em_percentage']
            avg_prompt_tokens = result['token_usage']['avg_prompt_tokens']
            avg_example_tokens = result['token_usage']['avg_example_tokens']
            
            if has_std and show_std:
                f1_std = result['answerable'].get('f1_std_percentage', 0.0)
                em_std = result['answerable'].get('em_std_percentage', 0.0)
                num_runs = result.get('num_runs', 1)
                print(f"{k:<4} | {f1_pct:>6.2f}±{f1_std:>4.2f}%  | {em_pct:>6.2f}±{em_std:>4.2f}%  | {avg_prompt_tokens:>8.1f} tok | {avg_example_tokens:>9.1f} tok | {num_runs:<6}")
            else:
                no_ans_acc = result['unanswerable']['no_answer_accuracy']
                ans_count = result['answerable']['count']
                unans_count = result['unanswerable']['count']
                print(f"{k:<4} | {f1_pct:>6.2f}%    | {em_pct:>6.2f}%    | {no_ans_acc:>8.2f}%    | {avg_prompt_tokens:>8.1f} tok | {avg_example_tokens:>9.1f} tok | {ans_count:<6} | {unans_count:<6}")
        
        # Find best k for this bin
        best_k_f1 = max(bin_data['k_results'].keys(), 
                       key=lambda k: bin_data['k_results'][k]['answerable']['avg_f1'])
        best_f1 = bin_data['k_results'][best_k_f1]['answerable']['f1_percentage']
        best_avg_example = bin_data['k_results'][best_k_f1]['token_usage']['avg_example_tokens']
        
        print(f"\n✓ Best k for Bin {bin_id}: k={best_k_f1} (F1: {best_f1:.2f}%, Avg Example: {best_avg_example:.1f} tokens)")
    
    # Cross-bin comparison at similar token budgets
    print(f"\n\n{'='*90}")
    print(f"Cross-Bin Comparison (Similar Token Budget)")
    print(f"{'='*90}")
    print(f"\nComparing performance when using similar number of tokens:\n")
    
    # Check if multi-run
    first_bin = list(evaluation['bins'].values())[0]
    first_result = list(first_bin['k_results'].values())[0]
    has_std = 'std_f1' in first_result['answerable']
    
    # Find k values that use similar token budgets across bins
    if has_std and show_std:
        print(f"{'Bin':<6} | {'k':<4} | {'Token Range':<14} | {'Avg F1':<15} | {'Avg EM':<15} | {'Avg Prompt':<12} | {'Avg Example':<13}")
    else:
        print(f"{'Bin':<6} | {'k':<4} | {'Token Range':<14} | {'Avg F1':<10} | {'Avg EM':<10} | {'Avg Prompt':<12} | {'Avg Example':<13}")
    print(f"{'-'*100}")
    
    for bin_id in sorted(evaluation['bins'].keys(), key=int):
        bin_data = evaluation['bins'][bin_id]
        bin_range = bin_data['bin_range']
        
        for k in sorted(bin_data['k_results'].keys(), key=int):
            result = bin_data['k_results'][k]
            f1_pct = result['answerable']['f1_percentage']
            em_pct = result['answerable']['em_percentage']
            avg_prompt_tokens = result['token_usage']['avg_prompt_tokens']
            avg_example_tokens = result['token_usage']['avg_example_tokens']
            
            if has_std and show_std:
                f1_std = result['answerable'].get('f1_std_percentage', 0.0)
                em_std = result['answerable'].get('em_std_percentage', 0.0)
                print(f"{bin_id:<6} | {k:<4} | [{bin_range[0]}-{bin_range[1]}]     | {f1_pct:>6.2f}±{f1_std:>4.2f}%  | {em_pct:>6.2f}±{em_std:>4.2f}%  | {avg_prompt_tokens:>8.1f} tok | {avg_example_tokens:>9.1f} tok")
            else:
                print(f"{bin_id:<6} | {k:<4} | [{bin_range[0]}-{bin_range[1]}]     | {f1_pct:>6.2f}%    | {em_pct:>6.2f}%    | {avg_prompt_tokens:>8.1f} tok | {avg_example_tokens:>9.1f} tok")
    
    # Overall summary
    print(f"\n\n{'='*80}")
    print(f"Summary: Key Findings")
    print(f"{'='*80}\n")
    
    # Find overall best configuration
    best_overall_bin = None
    best_overall_k = None
    best_overall_f1 = 0.0
    
    for bin_id in evaluation['bins'].keys():
        for k in evaluation['bins'][bin_id]['k_results'].keys():
            f1 = evaluation['bins'][bin_id]['k_results'][k]['answerable']['avg_f1']
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_bin = bin_id
                best_overall_k = k
    
    if best_overall_bin:
        best_result = evaluation['bins'][best_overall_bin]['k_results'][best_overall_k]
        bin_range = best_result['bin_range']
        avg_example_tokens = best_result['token_usage']['avg_example_tokens']
        print(f"✓ Best Overall Configuration:")
        print(f"  - Bin {best_overall_bin} (token length [{bin_range[0]}-{bin_range[1]}])")
        print(f"  - k = {best_overall_k} examples")
        print(f"  - F1: {best_result['answerable']['f1_percentage']:.2f}%")
        print(f"  - EM: {best_result['answerable']['em_percentage']:.2f}%")
        print(f"  - Avg Prompt Tokens: {best_result['token_usage']['avg_prompt_tokens']:.1f}")
        print(f"  - Avg Example Tokens: {avg_example_tokens:.1f}")
    
    # Per-bin best k
    print(f"\n✓ Best k per Bin:")
    for bin_id in sorted(evaluation['bins'].keys()):
        bin_data = evaluation['bins'][bin_id]
        best_k = max(bin_data['k_results'].keys(), 
                    key=lambda k: bin_data['k_results'][k]['answerable']['avg_f1'])
        best_result = bin_data['k_results'][best_k]
        bin_range = best_result['bin_range']
        avg_example_tokens = best_result['token_usage']['avg_example_tokens']
        print(f"  - Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: k={best_k} (F1: {best_result['answerable']['f1_percentage']:.2f}%, Avg Example: {avg_example_tokens:.1f} tokens)")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate length bin experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single run
  python evaluate_length_bin_results.py \\
      --results-file length_bin_experiments/google_gemma-3-4b-it_length_bin_budget_1000_run_1_seed_42_results.json
  
  # Evaluate and average multiple runs (automatically finds run_1, run_2, ..., run_N)
  python evaluate_length_bin_results.py \\
      --results-file length_bin_experiments/google_gemma-3-4b-it_length_bin_budget_1000_run_1_seed_42_results.json \\
      --num-runs 5
  
  # Evaluate multiple runs and save aggregated results
  python evaluate_length_bin_results.py \\
      --results-file length_bin_experiments/google_gemma-3-4b-it_length_bin_budget_1000_run_1_seed_42_results.json \\
      --num-runs 5 \\
      --output evaluation_averaged.json
  
  # Evaluate all results in directory
  python evaluate_length_bin_results.py \\
      --results-dir length_bin_experiments \\
      --pattern "*_length_bin_budget_*_results.json"
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-file",
        type=str,
        help="Path to a results JSON file (for single run or first run of multiple)"
    )
    group.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing results files"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs to average (default: 1). Automatically finds run_1, run_2, ..., run_N files"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_length_bin_budget_*_results.json",
        help="Glob pattern for results files (used with --results-dir)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save evaluation results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Collect results files
    if args.results_file:
        if args.num_runs > 1:
            # Find multiple run files based on the pattern
            base_path = Path(args.results_file)
            
            # Extract base pattern (remove run_X_seed_Y from filename)
            filename = base_path.name
            # Pattern: model_budget_run_X_seed_Y_results.json -> model_budget
            import re
            match = re.search(r'(.+)_run_\d+_seed_\d+_results\.json', filename)
            if match:
                base_name = match.group(1)
                result_files = []
                
                for run_idx in range(1, args.num_runs + 1):
                    # Try to find file with pattern: base_name_run_X_seed_*_results.json
                    pattern = base_path.parent / f"{base_name}_run_{run_idx}_seed_*_results.json"
                    matching_files = glob.glob(str(pattern))
                    
                    if matching_files:
                        result_files.append(matching_files[0])
                    else:
                        print(f"Warning: Could not find file for run {run_idx} matching pattern: {pattern}")
                
                if len(result_files) != args.num_runs:
                    print(f"\n❌ Error: Expected {args.num_runs} run files, found {len(result_files)}")
                    print(f"   Base pattern: {base_name}_run_X_seed_Y_results.json")
                    return
                
                print(f"\n{'='*80}")
                print(f"Found {len(result_files)} run files:")
                print(f"{'='*80}")
                for f in result_files:
                    print(f"  - {Path(f).name}")
                print(f"{'='*80}\n")
            else:
                print(f"\n❌ Error: Filename does not match expected pattern for multiple runs")
                print(f"   Expected: *_run_X_seed_Y_results.json")
                print(f"   Got: {filename}")
                return
        else:
            result_files = [args.results_file]
    else:
        pattern = os.path.join(args.results_dir, args.pattern)
        result_files = sorted(glob.glob(pattern))
        
        if not result_files:
            print(f"\n❌ No result files found matching pattern: {pattern}")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(result_files)} result files")
        print(f"{'='*80}")
        for f in result_files:
            print(f"  - {os.path.basename(f)}")
        print(f"{'='*80}\n")
    
    # Evaluate
    if args.num_runs > 1 and args.results_file:
        # Multiple runs: compute averaged results
        filename = f"averaged_{args.num_runs}_runs"
        
        print(f"\n{'='*80}")
        print(f"Evaluating averaged results across {args.num_runs} runs")
        print(f"{'='*80}")
        
        evaluation = evaluate_multiple_runs(result_files)
        
        # Print results
        print_evaluation(evaluation, show_std=True)
        
        all_evaluations = {filename: evaluation}
    else:
        # Single run or directory mode
        all_evaluations = {}
        
        for result_file in result_files:
            filename = os.path.basename(result_file)
            
            print(f"\n{'='*80}")
            print(f"Evaluating: {filename}")
            print(f"{'='*80}")
            
            # Evaluate
            evaluation = evaluate_length_bin_results(result_file, verbose=True)
            
            # Print results
            print_evaluation(evaluation, show_std=False)
            
            # Store evaluation
            all_evaluations[filename] = evaluation
    
    # Save to file if requested
    if args.output:
        output_data = {
            'evaluations': all_evaluations,
            'summary': {
                'total_files': len(all_evaluations),
                'files': list(all_evaluations.keys())
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Evaluation results saved to: {args.output}\n")


if __name__ == "__main__":
    main()

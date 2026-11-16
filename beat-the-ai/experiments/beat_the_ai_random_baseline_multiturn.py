#!/usr/bin/env python3
"""
Beat the AI Random Baseline Experiment - Multi-Turn Version

This script runs a RANDOM BASELINE for ICL performance comparison.
Instead of selecting examples from a specific length bin, it samples randomly
from the GLOBAL pool (all bins combined) until the token budget is exhausted.

Key design:
- Pool = all examples from all 4 bins merged together
- Per query: greedily sample random examples one by one, tracking cumulative
  token usage, stopping when the next example would exceed the token budget
- Token trace: cumulative ICL token count after each example is added, saved
  per query for later plotting
- Multi-turn format: identical to beat_the_ai_length_bin_experiments_multiturn.py
  (each ICL example is a separate user-assistant turn pair)
- 5 runs with different seeds, results averaged and saved

Output per run:
  <model>_random_baseline_budget_<budget>_multiturn_run_<n>_seed_<s>_results.json

Fields per query:
  - k_used: number of ICL examples that fit
  - cumulative_token_trace: [tokens_after_ex1, tokens_after_ex2, ...]
  - llm_response, is_impossible, answers_text, ...
"""

import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from elasticsearch import Elasticsearch
from vllm import LLM, SamplingParams
from more_itertools import chunked

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import ES_HOST, ES_PORT, ES_INDEX_NAME, ES_SCHEME

from transformers import AutoTokenizer

# Tokenizer will be initialized with the model
tokenizer = None

# Fixed instruction overhead reserved for the test query + formatting
INSTRUCTION_TOKEN_OVERHEAD = 140

# System prompt — contains full task instruction (applied once, not repeated per turn)
SYSTEM_PROMPT = """You are an information extraction system.

You must answer the QUESTION using ONLY the CONTEXT.

Task:
- If the CONTEXT contains an answer, extract it as a contiguous span copied verbatim from the CONTEXT.
- If the CONTEXT does not contain enough information to answer, output NO_ANSWER.

Rules:
1) The extracted ANSWER must be an EXACT contiguous substring from CONTEXT (verbatim, case-sensitive).
2) Prefer the minimal span that answers the question precisely and unambiguously.
3) If no such span exists in CONTEXT, output exactly: NO_ANSWER
4) Output exactly one line with this format and nothing else:
ANSWER: <answer span or NO_ANSWER>"""


def count_tokens(text: str) -> int:
    """Count tokens in text using the model's tokenizer."""
    global tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Call initialize_tokenizer() first.")
    return len(tokenizer.encode(text, add_special_tokens=False))


def initialize_tokenizer(model_name: str):
    """Initialize the tokenizer for the given model."""
    global tokenizer
    print(f"  Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  ✓ Tokenizer loaded")


class RandomBaselineRunnerMultiTurn:
    """
    Run LLM inference random baseline using multi-turn conversations.

    Examples are sampled uniformly at random from the global pool (all bins
    merged) and added greedily until the token budget is exhausted.
    """

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90,
                 token_budget: int = 5000,
                 max_model_len: Optional[int] = None):
        print(f"✓ Initializing Random Baseline Runner (Multi-Turn)...")
        print(f"  Model: {model}")
        print(f"  Token budget: {token_budget} tokens")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        if max_model_len:
            print(f"  Max model length: {max_model_len} tokens")

        initialize_tokenizer(model)

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.model = model
        self.token_budget = token_budget
        self.es = Elasticsearch([f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"],
                                verify_certs=False, request_timeout=60)

        # Global pool: populated by load_bin_examples
        self.all_examples: List[Dict] = []

        print(f"✓ Random Baseline Runner initialized (Multi-Turn mode)")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_bin_examples(self, bin_dir: str):
        """
        Load all bin examples and merge them into one global pool.

        Args:
            bin_dir: Directory containing bin_*_full.json files
        """
        print(f"\nLoading bin examples from: {bin_dir}")
        bin_dir_path = Path(bin_dir)

        summary_file = bin_dir_path / "bins_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        bin_counts = {}
        for bin_info in summary['bins']:
            bin_id = bin_info['bin_id']
            bin_range = bin_info['bin_range']

            full_file = bin_dir_path / f"bin_{bin_id}_{bin_range[0]}_{bin_range[1]}_full.json"
            with open(full_file, 'r') as f:
                bin_data = json.load(f)

            examples = bin_data['examples']
            # Tag each example with its source bin for logging
            for ex in examples:
                ex['_source_bin'] = bin_id
            self.all_examples.extend(examples)
            bin_counts[bin_id] = len(examples)

            print(f"  ✓ Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: {len(examples)} examples")

        print(f"✓ Global pool: {len(self.all_examples)} total examples from {len(bin_counts)} bins\n")

    def load_test_queries(self, test_file: str,
                          max_queries: Optional[int] = None) -> List[Dict]:
        """Load test queries from Beat the AI dev JSON file."""
        print(f"Loading test queries from: {test_file}")
        test_queries = []

        with open(test_file, 'r', encoding='utf-8') as f:
            beat_the_ai_data = json.load(f)

        for article in beat_the_ai_data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qa_id = qa['id']
                    question = qa['question']
                    is_impossible = qa.get('is_impossible', False)

                    answers_text = []
                    if not is_impossible and 'answers' in qa:
                        answers_text = [ans['text'] for ans in qa['answers']]

                    test_queries.append({
                        'qa_id': qa_id,
                        'title': title,
                        'question_text': question,
                        'context_text': context,
                        'is_impossible': is_impossible,
                        'answers_text': answers_text
                    })

                    if max_queries and len(test_queries) >= max_queries:
                        print(f"✓ Loaded {len(test_queries)} test queries (limited by max_queries)")
                        return test_queries

        print(f"✓ Loaded {len(test_queries)} test queries")
        return test_queries

    # ------------------------------------------------------------------
    # Prompt building (identical to multiturn experiment)
    # ------------------------------------------------------------------

    def build_user_turn(self, context: str, question: str,
                        include_instruction: bool = False) -> str:
        """
        Build a user turn with context and question.
        Optionally prepend SYSTEM_PROMPT to the first ICL turn.
        """
        body = f"""CONTEXT:
<<<
{context}
>>>

QUESTION:
{question}"""
        if include_instruction:
            return f"{SYSTEM_PROMPT}\n\n{body}"
        return body

    def build_assistant_turn(self, answer: str) -> str:
        """Build an assistant turn with the given answer."""
        return f"ANSWER: {answer}"

    def build_multiturn_conversation(self, test_question: str, test_context: str,
                                     few_shot_examples: List[Dict]) -> List[Dict[str, str]]:
        """
        Build a multi-turn conversation where each ICL example is a separate turn.
        First ICL user turn gets the task instruction prepended (same as multiturn experiment).
        """
        conversation = []

        for idx, example in enumerate(few_shot_examples):
            context = example.get('context_text', '')
            question = example.get('question_text', '')

            if example.get('is_impossible', False):
                answer = "NO_ANSWER"
            else:
                answer = example.get('answer_text', '').strip() or "NO_ANSWER"

            user_content = self.build_user_turn(context, question,
                                                include_instruction=(idx == 0))
            conversation.append({"role": "user", "content": user_content})
            conversation.append({"role": "assistant",
                                  "content": self.build_assistant_turn(answer)})

        # Final user turn: the actual test question (no instruction repeated)
        conversation.append({"role": "user",
                              "content": self.build_user_turn(test_context, test_question)})
        return conversation

    # ------------------------------------------------------------------
    # Random sampling with token budget
    # ------------------------------------------------------------------

    def sample_examples_for_budget(self, query_idx: int,
                                   run_seed: int) -> Tuple[List[Dict], List[int]]:
        """
        Randomly sample examples from the global pool until the token budget
        is exhausted.  The sampling order is randomized independently per
        (query, run) pair using seed = run_seed + query_idx * 31337.

        Budget accounting:
            available = token_budget - INSTRUCTION_TOKEN_OVERHEAD
                        - count_tokens(SYSTEM_PROMPT)   ← one-time first-turn cost
            For each candidate example (in shuffle order):
                cost = count_tokens(user_turn_body) + count_tokens(assistant_turn)
                       (instruction tokens NOT added here — already charged above)
            Greedy: add while cumulative_cost + cost <= available

        Args:
            query_idx: Index of the test query (makes the seed query-specific)
            run_seed: Base seed for this run

        Returns:
            (selected_examples, cumulative_token_trace)
            cumulative_token_trace[i] = total ICL cost after adding example i
        """
        query_seed = run_seed + query_idx * 31337
        rng = random.Random(query_seed)
        shuffled = rng.sample(self.all_examples, len(self.all_examples))

        system_prompt_tokens = count_tokens(SYSTEM_PROMPT)
        available = self.token_budget - INSTRUCTION_TOKEN_OVERHEAD - system_prompt_tokens

        selected: List[Dict] = []
        token_trace: List[int] = []
        cumulative = 0

        for example in shuffled:
            context = example.get('context_text', '')
            question = example.get('question_text', '')
            if example.get('is_impossible', False):
                answer = "NO_ANSWER"
            else:
                answer = example.get('answer_text', '').strip() or "NO_ANSWER"

            # Cost = context+question body + answer (instruction is one-time, already charged)
            user_body = self.build_user_turn(context, question, include_instruction=False)
            assistant_body = self.build_assistant_turn(answer)
            cost = count_tokens(user_body) + count_tokens(assistant_body)

            if cumulative + cost > available:
                break

            cumulative += cost
            selected.append(example)
            token_trace.append(cumulative)

        return selected, token_trace

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference_batch(self, conversations: List[List[Dict[str, str]]],
                            temperature: float = 0.0,
                            top_p: float = 1.0,
                            max_tokens: int = 128) -> List[str]:
        """Run batched inference on multiple multi-turn conversations."""
        conversations_with_system = [
            [{"role": "system", "content": SYSTEM_PROMPT}] + conv
            for conv in conversations
        ]

        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        try:
            outputs = self.llm.chat(conversations_with_system,
                                    sampling_params=sampling)
            return [out.outputs[0].text.strip() for out in outputs]
        except Exception as e:
            print(f"\n✗ Error during inference: {e}")
            return ["ERROR"] * len(conversations)

    # ------------------------------------------------------------------
    # Prompt samples
    # ------------------------------------------------------------------

    def save_prompt_samples(self, query_sequences: List[Dict], n_queries: int,
                             sample_dir: str, model_name: str) -> str:
        """
        Save the k-progression for the first n_queries as human-readable text.
        Each query shows the conversation at k=1, k=2, ... k_max to illustrate
        how examples are added incrementally.
        """
        os.makedirs(sample_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(
            sample_dir, f"{model_name}_random_baseline_samples_{timestamp}.txt")

        samples = query_sequences[:n_queries]
        lines = []
        lines.append(f"Random Baseline Prompt Samples — {model_name}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Showing k-progression for first {len(samples)} queries")
        lines.append(f"System prompt:\n{SYSTEM_PROMPT}")
        lines.append("")

        for q_idx, seq in enumerate(samples, 1):
            qa_id = seq['qa_id']
            selected = seq['selected_examples']
            token_trace = seq['token_trace']
            query_data = seq['query_data']
            k_max = len(selected)

            sep = "=" * 80
            lines.append(sep)
            lines.append(f"QUERY {q_idx}/{len(samples)}  —  QA ID: {qa_id}")
            lines.append(f"  k_max: {k_max}  |  token_trace: {token_trace}")
            lines.append(sep)
            lines.append("")

            for k in range(1, k_max + 1):
                conversation = self.build_multiturn_conversation(
                    query_data['question_text'],
                    query_data['context_text'],
                    selected[:k]
                )
                lines.append(f"  ── k={k}  cumulative_tokens={token_trace[k-1]} ──")
                for i, turn in enumerate(conversation):
                    if i < len(conversation) - 1:
                        if turn['role'] == 'user':
                            lines.append(f"    [USER — ICL EXAMPLE {i // 2 + 1}]")
                        else:
                            lines.append(f"    [ASSISTANT — ICL EXAMPLE {i // 2 + 1}]")
                    else:
                        lines.append("    [USER — TEST QUERY]")
                    lines.append(turn['content'])
                    lines.append("")
                lines.append("")

        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"  ✓ Prompt samples saved to: {sample_path}")
        return sample_path

    # ------------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------------

    def run_experiments(self, test_queries_file: str,
                        bin_dir: str,
                        temperature: float = 0.0,
                        top_p: float = 1.0,
                        max_tokens: int = 128,
                        batch_size: int = 8,
                        max_test_queries: Optional[int] = None,
                        output_dir: str = "random_baseline_multiturn",
                        seed: int = 42,
                        num_runs: int = 5,
                        prompt_sample_dir: Optional[str] = None) -> List[str]:
        """Run the complete random baseline experiment across num_runs seeds."""
        print(f"\n{'='*70}")
        print(f"Random Baseline Experiment (MULTI-TURN) with Fixed Token Budget")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Token budget: {self.token_budget} tokens")
        print(f"Base random seed: {seed}")
        print(f"Number of runs: {num_runs}")
        print(f"{'='*70}\n")

        self.load_bin_examples(bin_dir)
        test_queries = self.load_test_queries(test_queries_file, max_test_queries)
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.replace(':', '_').replace('/', '_')

        system_prompt_tokens = count_tokens(SYSTEM_PROMPT)
        available_for_icl = self.token_budget - INSTRUCTION_TOKEN_OVERHEAD - system_prompt_tokens
        print(f"Fixed instruction overhead: {INSTRUCTION_TOKEN_OVERHEAD} tokens")
        print(f"System prompt (one-time first-turn cost): {system_prompt_tokens} tokens")
        print(f"Available for ICL examples: {available_for_icl} tokens")
        print(f"Global ICL pool size: {len(self.all_examples)} examples\n")

        output_paths = []
        for run_idx in range(num_runs):
            run_seed = seed + run_idx
            print(f"\n{'='*70}")
            print(f"RUN {run_idx + 1}/{num_runs} (seed={run_seed})")
            print(f"{'='*70}\n")

            output_path = self._run_single_experiment(
                test_queries=test_queries,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                batch_size=batch_size,
                output_dir=output_dir,
                model_name=model_name,
                run_seed=run_seed,
                run_idx=run_idx,
                num_runs=num_runs,
                prompt_sample_dir=prompt_sample_dir if run_idx == 0 else None
            )
            output_paths.append(output_path)

        print(f"\n{'='*70}")
        print(f"✓ All {num_runs} runs complete!")
        print(f"{'='*70}")
        print("Results saved to:")
        for i, path in enumerate(output_paths, 1):
            print(f"  Run {i}: {path}")
        print(f"{'='*70}\n")

        return output_paths

    def _run_single_experiment(self, test_queries: List[Dict],
                               temperature: float,
                               top_p: float,
                               max_tokens: int,
                               batch_size: int,
                               output_dir: str,
                               model_name: str,
                               run_seed: int,
                               run_idx: int,
                               num_runs: int,
                               prompt_sample_dir: Optional[str] = None) -> str:
        """Run a single experiment with a specific seed.

        For each query we sample a shuffled sequence of examples (greedy budget
        fill) and then build k_max separate conversations — one for each prefix
        length k=1..k_max.  Every (query, k) pair gets its own inference call,
        allowing per-k F1 curves to be computed during evaluation.
        """
        if num_runs > 1:
            output_path = os.path.join(
                output_dir,
                f"{model_name}_random_baseline_budget_{self.token_budget}"
                f"_multiturn_run_{run_idx + 1}_seed_{run_seed}_results.json"
            )
        else:
            output_path = os.path.join(
                output_dir,
                f"{model_name}_random_baseline_budget_{self.token_budget}_multiturn_results.json"
            )

        # ── Step 1: sample full sequences for every query ──────────────────
        print("Sampling example sequences for all queries...")
        query_sequences: List[Dict] = []
        for query_idx, query_data in enumerate(
                tqdm(test_queries, desc="Sampling sequences")):
            selected, token_trace = self.sample_examples_for_budget(
                query_idx, run_seed)
            query_sequences.append({
                'qa_id': query_data['qa_id'],
                'query_data': query_data,
                'selected_examples': selected,
                'token_trace': token_trace,
            })

        k_max_values = [len(s['selected_examples']) for s in query_sequences]
        print(f"✓ Sequences sampled for {len(query_sequences)} queries")
        print(f"  k_max — min: {min(k_max_values)}, max: {max(k_max_values)}, "
              f"mean: {np.mean(k_max_values):.1f}, "
              f"median: {np.median(k_max_values):.1f}\n")

        # Save prompt samples (first run only, shows k-progression per query)
        if prompt_sample_dir is not None:
            print("Saving prompt samples for manual inspection...")
            self.save_prompt_samples(query_sequences, 5,
                                     prompt_sample_dir, model_name)
            print("")

        # ── Step 2: build one conversation per (query, k) ──────────────────
        print("Building multi-turn conversations for all (query, k) pairs...")
        all_conv_data: List[Dict] = []
        for seq in tqdm(query_sequences, desc="Building conversations"):
            query_data = seq['query_data']
            selected = seq['selected_examples']
            token_trace = seq['token_trace']
            for k in range(1, len(selected) + 1):
                conversation = self.build_multiturn_conversation(
                    query_data['question_text'],
                    query_data['context_text'],
                    selected[:k]
                )
                all_conv_data.append({
                    'qa_id': query_data['qa_id'],
                    'question': query_data['question_text'],
                    'is_impossible': query_data['is_impossible'],
                    'answers_text': query_data['answers_text'],
                    'k': k,
                    'cumulative_tokens': token_trace[k - 1],
                    'conversation': conversation,
                })

        print(f"✓ Built {len(all_conv_data)} conversations "
              f"({len(test_queries)} queries × avg {np.mean(k_max_values):.1f} k-steps)\n")

        # ── Step 3: batched inference ───────────────────────────────────────
        print(f"Running batched inference (batch_size={batch_size})...")
        all_answers: List[str] = []
        for batch_data in tqdm(list(chunked(all_conv_data, batch_size)),
                               desc="Processing batches"):
            batch_conversations = [d['conversation'] for d in batch_data]
            batch_answers = self.run_inference_batch(
                batch_conversations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            all_answers.extend(batch_answers)
        print(f"✓ Inference complete\n")

        # ── Step 4: group results by k ──────────────────────────────────────
        print("Organising results by k...")
        by_k: Dict[int, List[Dict]] = defaultdict(list)
        for i, conv_data in enumerate(all_conv_data):
            by_k[conv_data['k']].append({
                'qa_id': conv_data['qa_id'],
                'question': conv_data['question'],
                'is_impossible': conv_data['is_impossible'],
                'answers_text': conv_data['answers_text'],
                'cumulative_tokens': conv_data['cumulative_tokens'],
                'llm_response': all_answers[i],
            })

        system_prompt_tokens = count_tokens(SYSTEM_PROMPT)
        results = {
            "metadata": {
                "model": self.model,
                "token_budget": self.token_budget,
                "instruction_overhead": INSTRUCTION_TOKEN_OVERHEAD,
                "system_prompt_tokens": system_prompt_tokens,
                "available_for_icl": (self.token_budget
                                      - INSTRUCTION_TOKEN_OVERHEAD
                                      - system_prompt_tokens),
                "pool_size": len(self.all_examples),
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "seed": run_seed,
                "run_number": run_idx + 1,
                "total_runs": num_runs,
                "total_queries": len(test_queries),
                "total_inference_calls": len(all_conv_data),
                "approach": "random-baseline-multi-turn-per-k",
                "timestamp": datetime.now().isoformat(),
                "k_max_stats": {
                    "min": int(min(k_max_values)),
                    "max": int(max(k_max_values)),
                    "mean": float(np.mean(k_max_values)),
                    "median": float(np.median(k_max_values)),
                    "std": float(np.std(k_max_values)),
                },
            },
            # Each entry: one k-value with all query responses for that k.
            # num_queries decreases as k increases (only queries with k_max >= k).
            "experiments": [
                {
                    "k": k,
                    "num_queries": len(by_k[k]),
                    "responses": by_k[k],
                }
                for k in sorted(by_k.keys())
            ],
        }

        print(f"Writing results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Run {run_idx + 1}/{num_runs} complete!")
        print(f"  Seed: {run_seed}  |  Queries: {len(test_queries)}  "
              f"|  k range: 1–{max(k_max_values)}  "
              f"|  Total calls: {len(all_conv_data)}\n")

        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Random baseline ICL experiment — multi-turn, global pool, token-budget greedy sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with default settings
  python beat_the_ai_random_baseline_multiturn.py \\
      --test-queries /path/to/dev.json \\
      --bin-dir beat_the_ai_code/get_token_length_size/bin_examples

  # Quick sanity check
  python beat_the_ai_random_baseline_multiturn.py \\
      --test-queries /path/to/dev.json \\
      --bin-dir beat_the_ai_code/get_token_length_size/bin_examples \\
      --max-test-queries 20 --num-runs 1
        """
    )
    parser.add_argument("--test-queries", type=str, required=True,
                        help="Path to Beat the AI dev JSON file")
    parser.add_argument("--bin-dir", type=str, required=True,
                        help="Directory with extracted bin examples (all bins pooled)")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="Single model name or path (ignored if --models is set)")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="One or more model names/paths to run sequentially")
    parser.add_argument("--token-budget", type=int, default=5000,
                        help="Token budget for ICL context (default: 5000)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling (default: 1.0)")
    parser.add_argument("--max-tokens", type=int, default=800,
                        help="Max tokens to generate (default: 800)")
    parser.add_argument("--batch-size", type=int, default=70,
                        help="Batch size for inference (default: 70)")
    parser.add_argument("--max-test-queries", type=int, default=None,
                        help="Max test queries to process (default: all)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU memory utilization (default: 0.90)")
    parser.add_argument("--max-model-len", type=int, default=10000,
                        help="Maximum model context length (default: 10000)")
    parser.add_argument("--output-dir", type=str,
                        default="beat_the_ai_code/get_token_length_size/random_baseline_multiturn",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs with different seeds (default: 5)")
    parser.add_argument("--prompt-sample-dir", type=str, default=None,
                        help="If set, save the first 5 conversations here (run 1 only)")

    args = parser.parse_args()

    models_to_run = args.models if args.models else [args.model]

    print(f"Models to run ({len(models_to_run)}): {models_to_run}")

    for model_idx, model_name in enumerate(models_to_run):
        print(f"\n{'#'*70}")
        print(f"MODEL {model_idx + 1}/{len(models_to_run)}: {model_name}")
        print(f"{'#'*70}\n")

        # Reset seeds for each model so runs are reproducible independently
        random.seed(args.seed)
        np.random.seed(args.seed)

        runner = RandomBaselineRunnerMultiTurn(
            model=model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            token_budget=args.token_budget,
            max_model_len=args.max_model_len
        )

        runner.run_experiments(
            test_queries_file=args.test_queries,
            bin_dir=args.bin_dir,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_test_queries=args.max_test_queries,
            output_dir=args.output_dir,
            seed=args.seed,
            num_runs=args.num_runs,
            prompt_sample_dir=args.prompt_sample_dir
        )

        # Release GPU memory before loading the next model
        if model_idx < len(models_to_run) - 1:
            print(f"\nReleasing GPU memory before next model...")
            del runner
            torch.cuda.empty_cache()
            print("✓ GPU memory released\n")

    print(f"\n{'#'*70}")
    print(f"All {len(models_to_run)} model(s) complete.")
    print(f"{'#'*70}")

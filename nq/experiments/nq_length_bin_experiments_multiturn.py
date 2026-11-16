#!/usr/bin/env python3
"""
NQ Length-Based Bin Experiments with Fixed Token Budget - Multi-Turn Version

This script runs experiments to test the impact of example token length on ICL performance
for Natural Questions using a MULTI-TURN conversational approach where each ICL example
is presented as a separate conversation turn.

Key differences from QASPER version:
- Context is a Wikipedia document (document_text)
- Answers are extractive short spans, long answers, or unanswerable
- ICL examples come from Elasticsearch bins (pre-extracted)
- Test queries come from a JSONL dev file with token-offset annotations
"""

import json
import os
import random
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from more_itertools import chunked
from transformers import AutoTokenizer

# Tokenizer will be initialized with the model
tokenizer = None

# Fixed instruction overhead (instruction + test query formatting)
INSTRUCTION_TOKEN_OVERHEAD = 140

NQ_DEV_PATH = "/mnt/gpu-fastdata/anxo/sigir26-short-QA-ICL/nq-dataset-format/dataset-dev-example/bin1-simplified-nq-dev.jsonl"

# System prompt and initial instruction (identical for NQ)
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

INITIAL_INSTRUCTION = SYSTEM_PROMPT


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


class NQLengthBinExperimentRunnerMultiTurn:
    """Run LLM inference experiments with token length bins on NQ using multi-turn conversations."""

    def __init__(self, model: str = "google/gemma-3-4b-it",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90,
                 token_budget: int = 5000,
                 max_model_len: Optional[int] = None):
        print(f"✓ Initializing NQ Length Bin Experiment Runner (Multi-Turn)...")
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
        self.bin_examples = {}

        print(f"✓ Experiment runner initialized (Multi-Turn mode)")

    def load_bin_examples(self, bin_dir: str):
        """Load pre-extracted bin examples from JSON files."""
        print(f"\nLoading bin examples from: {bin_dir}")
        bin_dir_path = Path(bin_dir)

        summary_file = bin_dir_path / "bins_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        for bin_info in summary['bins']:
            bin_id = bin_info['bin_id']
            bin_range = bin_info['bin_range']

            full_file = bin_dir_path / f"bin_{bin_id}_{bin_range[0]}_{bin_range[1]}_full.json"
            with open(full_file, 'r') as f:
                bin_data = json.load(f)

            self.bin_examples[bin_id] = {
                'range': bin_range,
                'examples': bin_data['examples']
            }
            print(f"  ✓ Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: {len(bin_data['examples'])} examples")

        print(f"✓ Loaded {len(self.bin_examples)} bins\n")

    def load_test_queries(self, test_file: str, max_queries: Optional[int] = None) -> List[Dict]:
        """
        Load test queries from an NQ dev JSONL file.

        Each line is one example with question_text, document_text, example_id,
        and annotations (with token-offset gold answers).
        """
        print(f"Loading test queries from: {test_file}")

        test_queries = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)

                question = doc.get("question_text", "")
                context = doc.get("document_text", "")
                example_id = doc.get("example_id", "")

                test_queries.append({
                    'example_id': example_id,
                    'question_text': question,
                    'context_text': context,
                })

                if max_queries and len(test_queries) >= max_queries:
                    print(f"✓ Loaded {len(test_queries)} test queries (limited by max_queries)")
                    return test_queries

        print(f"✓ Loaded {len(test_queries)} test queries")
        return test_queries

    def build_user_turn(self, context: str, question: str,
                        include_instruction: bool = False) -> str:
        """Build a user turn with context and question."""
        body = f"""CONTEXT:
<<<
{context}
>>>

QUESTION:
{question}"""
        if include_instruction:
            return f"{INITIAL_INSTRUCTION}\n\n{body}"
        return body

    def build_assistant_turn(self, answer: str) -> str:
        """Build an assistant turn with the answer."""
        return f"ANSWER: {answer}"

    def calculate_max_k_for_bin(self, bin_id: int) -> int:
        """
        Calculate maximum k value for a bin given the token budget.

        Uses the actual median token length of examples in the bin.

        Budget equation:
            first_turn_instruction + k * median_context <= token_budget - INSTRUCTION_TOKEN_OVERHEAD
        """
        examples = self.bin_examples[bin_id]['examples']
        lengths = sorted(ex['token_length'] for ex in examples)
        n = len(lengths)
        if n % 2 == 1:
            median_tokens = lengths[n // 2]
        else:
            median_tokens = (lengths[n // 2 - 1] + lengths[n // 2]) / 2

        first_turn_instruction_tokens = count_tokens(SYSTEM_PROMPT)
        available_tokens = self.token_budget - INSTRUCTION_TOKEN_OVERHEAD - first_turn_instruction_tokens
        max_k = int(available_tokens / median_tokens)

        # Store for logging
        self.bin_examples[bin_id]['median_tokens'] = median_tokens

        return max(1, max_k)

    def sample_examples_from_bin(self, bin_id: int, k: int, seed: int = 42) -> List[Dict]:
        """Sample k random examples from a specific bin."""
        examples = self.bin_examples[bin_id]['examples']

        if k > len(examples):
            print(f"  Warning: Requested k={k} but bin {bin_id} only has {len(examples)} examples")
            k = len(examples)

        random.seed(seed)
        return random.sample(examples, k)

    def build_multiturn_conversation(self, test_question: str, test_context: str,
                                     few_shot_examples: List[Dict]) -> List[Dict[str, str]]:
        """
        Build a multi-turn conversation where each ICL example is a separate turn.

        Returns:
            List of conversation turns in OpenAI format
        """
        conversation = []

        for idx, example in enumerate(few_shot_examples):
            # NQ bin examples use document_text and answer_text
            context = example.get('document_text', '')
            question = example.get('question_text', '')
            answer = example.get('answer_text', '').strip()
            if not answer:
                answer = "NO_ANSWER"

            user_content = self.build_user_turn(context, question,
                                                include_instruction=(idx == 0))
            conversation.append({"role": "user", "content": user_content})

            assistant_content = self.build_assistant_turn(answer)
            conversation.append({"role": "assistant", "content": assistant_content})

        # Final user turn: the actual test question
        test_user_content = self.build_user_turn(test_context, test_question)
        conversation.append({"role": "user", "content": test_user_content})

        return conversation

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
            outputs = self.llm.chat(
                conversations_with_system,
                sampling_params=sampling
            )
            return [out.outputs[0].text.strip() for out in outputs]
        except Exception as e:
            print(f"\n✗ Error during inference: {e}")
            return ["ERROR"] * len(conversations)

    def count_conversation_tokens(self, conversation: List[Dict[str, str]]) -> int:
        """Count total tokens in a multi-turn conversation."""
        return sum(count_tokens(turn['content']) for turn in conversation)

    def save_prompt_samples(self, conversations_data: List[Dict], n: int,
                            sample_dir: str, model_name: str) -> str:
        """Save the first n conversations as human-readable text for manual inspection."""
        os.makedirs(sample_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(sample_dir,
                                   f"{model_name}_nq_prompt_samples_{timestamp}.txt")

        samples = conversations_data[:n]
        lines = [
            f"Prompt samples — {model_name} (NQ)",
            f"Generated: {datetime.now().isoformat()}",
            f"Showing first {len(samples)} conversations",
            f"System prompt: {SYSTEM_PROMPT}",
            "",
        ]

        for idx, conv_data in enumerate(samples, 1):
            sep = "=" * 80
            lines.append(sep)
            lines.append(f"SAMPLE {idx}/{len(samples)}")
            lines.append(f"  Bin: {conv_data['bin_id']},  k: {conv_data['k']}")
            lines.append(f"  Example ID: {conv_data['example_id']}")
            lines.append(f"  Conversation tokens (excl. system): {conv_data['conversation_tokens']}")
            lines.append(f"  Number of turns: {conv_data['num_turns']}")
            lines.append(sep)
            lines.append("")

            conversation = conv_data['conversation']
            icl_pair = 0
            for i, turn in enumerate(conversation):
                if i < len(conversation) - 1:
                    if turn['role'] == 'user':
                        icl_pair += 1
                        lines.append(f"--- TURN {icl_pair} [USER — ICL EXAMPLE {icl_pair}] ---")
                    else:
                        lines.append(f"--- TURN {icl_pair} [ASSISTANT — ICL EXAMPLE {icl_pair}] ---")
                else:
                    lines.append("--- FINAL USER TURN [TEST QUERY] ---")
                lines.append(turn['content'])
                lines.append("")

        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"  ✓ Prompt samples saved to: {sample_path}")
        return sample_path

    def run_experiments(self, test_queries_file: str,
                       bin_dir: str,
                       temperature: float = 0.0,
                       top_p: float = 1.0,
                       max_tokens: int = 128,
                       batch_size: int = 8,
                       max_test_queries: Optional[int] = None,
                       output_dir: str = "nq_length_bin_experiments_multiturn",
                       seed: int = 42,
                       num_runs: int = 5,
                       prompt_sample_dir: Optional[str] = None) -> List[str]:
        """Run the complete length bin experiments with multi-turn conversations."""
        print(f"\n{'='*70}")
        print(f"NQ Length-Based Bin Experiments (MULTI-TURN) with Fixed Token Budget")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Token budget: {self.token_budget} tokens")
        print(f"Temperature: {temperature}")
        print(f"Batch size: {batch_size}")
        print(f"Base random seed: {seed}")
        print(f"Number of runs: {num_runs}")
        print(f"{'='*70}\n")

        self.load_bin_examples(bin_dir)
        test_queries = self.load_test_queries(test_queries_file, max_test_queries)

        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.replace(':', '_').replace('/', '_')

        # Calculate max k for each bin
        first_turn_instruction_tokens = count_tokens(SYSTEM_PROMPT)
        print(f"Using fixed instruction overhead: {INSTRUCTION_TOKEN_OVERHEAD} tokens")
        print(f"Task instruction: in system prompt + first ICL user turn (one-time, {first_turn_instruction_tokens} tok)")
        print(f"Available tokens for ICL examples: {self.token_budget - INSTRUCTION_TOKEN_OVERHEAD - first_turn_instruction_tokens}\n")

        print(f"Calculating max k for each bin (using median token length):")
        bin_k_ranges = {}
        for bin_id in sorted(self.bin_examples.keys()):
            max_k = self.calculate_max_k_for_bin(bin_id)
            bin_k_ranges[bin_id] = list(range(1, max_k + 1))
            bin_range = self.bin_examples[bin_id]['range']
            median_tok = self.bin_examples[bin_id]['median_tokens']
            print(f"  Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: median={median_tok:.0f} tok/example → k=1 to k={max_k}")

        # Prepare all experiment configurations
        all_experiments = []
        for bin_id in sorted(self.bin_examples.keys()):
            for k in bin_k_ranges[bin_id]:
                all_experiments.append({'bin_id': bin_id, 'k': k})

        print(f"\nTotal experiment configurations: {len(all_experiments)}")
        print(f"Total prompts per run: {len(all_experiments) * len(test_queries)}")
        print(f"Total prompts across {num_runs} runs: {len(all_experiments) * len(test_queries) * num_runs}\n")

        output_paths = []
        for run_idx in range(num_runs):
            run_seed = seed + run_idx

            print(f"\n{'='*70}")
            print(f"RUN {run_idx + 1}/{num_runs} (seed={run_seed})")
            print(f"{'='*70}\n")

            output_path = self._run_single_experiment(
                test_queries=test_queries,
                all_experiments=all_experiments,
                bin_k_ranges=bin_k_ranges,
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
        print(f"Results saved to:")
        for i, path in enumerate(output_paths, 1):
            print(f"  Run {i}: {path}")
        print(f"{'='*70}\n")

        return output_paths

    def _run_single_experiment(self, test_queries: List[Dict],
                               all_experiments: List[Dict],
                               bin_k_ranges: Dict,
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

        Memory-efficient: processes one experiment config (bin_id, k) at a time.
        """
        if num_runs > 1:
            output_path = os.path.join(
                output_dir,
                f"{model_name}_nq_length_bin_budget_{self.token_budget}_multiturn_run_{run_idx + 1}_seed_{run_seed}_results.json"
            )
        else:
            output_path = os.path.join(
                output_dir,
                f"{model_name}_nq_length_bin_budget_{self.token_budget}_multiturn_results.json"
            )

        # Prepare result structure
        results = {
            "metadata": {
                "dataset": "nq",
                "model": self.model,
                "token_budget": self.token_budget,
                "instruction_overhead": INSTRUCTION_TOKEN_OVERHEAD,
                "available_for_examples": self.token_budget - INSTRUCTION_TOKEN_OVERHEAD,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "seed": run_seed,
                "run_number": run_idx + 1,
                "total_runs": num_runs,
                "total_queries": len(test_queries),
                "approach": "multi-turn",
                "timestamp": datetime.now().isoformat(),
                "bins": {
                    str(bin_id): {
                        "range": self.bin_examples[bin_id]['range'],
                        "k_values": bin_k_ranges[bin_id]
                    }
                    for bin_id in sorted(self.bin_examples.keys())
                }
            },
            "experiments": []
        }

        total_processed = 0
        saved_prompt_samples = False

        # Process one experiment config at a time to keep memory bounded
        for exp_idx, exp_config in enumerate(tqdm(all_experiments, desc="Experiment configs")):
            bin_id = exp_config['bin_id']
            k = exp_config['k']

            # 1) Build conversations for this config only
            config_conversations = []
            config_metadata = []

            few_shot_examples = self.sample_examples_from_bin(bin_id, k, run_seed)

            for query_data in test_queries:
                conversation = self.build_multiturn_conversation(
                    query_data['question_text'],
                    query_data['context_text'],
                    few_shot_examples
                )

                conversation_tokens = self.count_conversation_tokens(conversation)

                # Calculate example token lengths (each example = user + assistant turn)
                example_token_lengths = []
                for i in range(0, len(conversation) - 1, 2):
                    if i + 1 < len(conversation):
                        user_tokens = count_tokens(conversation[i]['content'])
                        assistant_tokens = count_tokens(conversation[i + 1]['content'])
                        example_token_lengths.append(user_tokens + assistant_tokens)

                config_conversations.append(conversation)
                config_metadata.append({
                    'bin_id': bin_id,
                    'k': k,
                    'example_id': query_data['example_id'],
                    'question': query_data['question_text'],
                    'conversation_tokens': conversation_tokens,
                    'example_token_lengths': example_token_lengths,
                    'num_turns': len(conversation)
                })

            # Save prompt samples on the first config (first run only)
            if prompt_sample_dir is not None and not saved_prompt_samples:
                sample_data = [
                    {**meta, 'conversation': conv}
                    for meta, conv in zip(config_metadata[:5], config_conversations[:5])
                ]
                print("Saving prompt samples for manual inspection...")
                self.save_prompt_samples(sample_data, 5, prompt_sample_dir, model_name)
                saved_prompt_samples = True

            # 2) Run batched inference for this config
            config_answers = []
            for batch_convs in chunked(config_conversations, batch_size):
                batch_answers = self.run_inference_batch(
                    list(batch_convs),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                config_answers.extend(batch_answers)

            # 3) Collect results (conversations are NOT stored — only metadata + response)
            exp_results = {
                "bin_id": bin_id,
                "bin_range": self.bin_examples[bin_id]['range'],
                "k": k,
                "responses": []
            }
            for meta, answer in zip(config_metadata, config_answers):
                exp_results["responses"].append({
                    "example_id": meta['example_id'],
                    "question": meta['question'],
                    "conversation_tokens": meta['conversation_tokens'],
                    "example_token_lengths": meta['example_token_lengths'],
                    "num_turns": meta['num_turns'],
                    "llm_response": answer
                })

            results["experiments"].append(exp_results)
            total_processed += len(config_conversations)

            # Write results to disk after every config so partial results survive crashes
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Conversations are freed here when config_conversations goes out of scope

        print(f"\n✓ Run {run_idx + 1}/{num_runs} complete!")
        print(f"  Seed: {run_seed}")
        print(f"  Results saved to: {output_path}")
        print(f"  Total conversations processed: {total_processed}\n")

        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run NQ length-based bin experiments with fixed token budget (MULTI-TURN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python nq_length_bin_experiments_multiturn.py \\
      --test-queries /path/to/nq-dev.jsonl \\
      --bin-dir NQ_code/get_token_length_size/bin_examples

  # Custom token budget and fewer runs
  python nq_length_bin_experiments_multiturn.py \\
      --test-queries /path/to/nq-dev.jsonl \\
      --bin-dir NQ_code/get_token_length_size/bin_examples \\
      --token-budget 10000 --num-runs 1

  # Quick test
  python nq_length_bin_experiments_multiturn.py \\
      --test-queries /path/to/nq-dev.jsonl \\
      --bin-dir NQ_code/get_token_length_size/bin_examples \\
      --max-test-queries 20 --num-runs 1
        """
    )
    parser.add_argument("--test-queries", type=str, default=NQ_DEV_PATH,
                        help=f"Path to NQ dev JSONL file (default: {NQ_DEV_PATH})")
    parser.add_argument("--bin-dir", type=str, required=True,
                        help="Directory with extracted bin examples")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="Model name or path (default: google/gemma-3-4b-it)")
    parser.add_argument("--token-budget", type=int, default=5000,
                        help="Token budget for context (default: 5000)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling (default: 1.0)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate (default: 128)")
    parser.add_argument("--batch-size", type=int, default=60,
                        help="Batch size for inference (default: 60)")
    parser.add_argument("--max-test-queries", type=int, default=None,
                        help="Max test queries to process (default: all)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="GPU memory utilization (default: 0.90)")
    parser.add_argument("--max-model-len", type=int, default=65536,
                        help="Maximum model context length (default: 65536)")
    parser.add_argument("--output-dir", type=str, default="nq_length_bin_experiments_multiturn",
                        help="Output directory (default: nq_length_bin_experiments_multiturn)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Number of runs with different seeds (default: 5)")
    parser.add_argument("--prompt-sample-dir", type=str, default=None,
                        help="Save first 5 conversations as readable text (run 1 only)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    runner = NQLengthBinExperimentRunnerMultiTurn(
        model=args.model,
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

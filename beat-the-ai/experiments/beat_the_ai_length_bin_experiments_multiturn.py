#!/usr/bin/env python3
"""
Beat the AI Length-Based Bin Experiments with Fixed Token Budget - Multi-Turn Version

This script runs experiments to test the impact of example token length on ICL performance
using a MULTI-TURN conversational approach where each ICL example is presented as a separate
conversation turn (user shows example → assistant responds → repeat).

Key differences from single-turn version:
- Each ICL example becomes a user-assistant conversation turn
- Final turn is the test question that needs to be answered
- Same token budget and bin-based sampling logic
- Tests if multi-turn presentation improves learning from examples
"""

import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
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

# Fixed instruction overhead (instruction + test query formatting)
# This is reserved from the total budget and never used for examples
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


class LengthBinExperimentRunnerMultiTurn:
    """Run LLM inference experiments with token length bins using multi-turn conversations."""
    
    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90,
                 token_budget: int = 1000,
                 max_model_len: Optional[int] = None):
        """
        Initialize the experiment runner.
        
        Args:
            model: HuggingFace model id or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            token_budget: Maximum token budget for context (default: 1000)
            max_model_len: Maximum model context length (default: None, uses model's default)
        """
        print(f"✓ Initializing Length Bin Experiment Runner (Multi-Turn)...")
        print(f"  Model: {model}")
        print(f"  Token budget: {token_budget} tokens")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        if max_model_len:
            print(f"  Max model length: {max_model_len} tokens")
        
        # Initialize tokenizer for this model
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
        
        # Load bin examples from extracted files
        self.bin_examples = {}
        
        print(f"✓ Experiment runner initialized (Multi-Turn mode)")
    
    def load_bin_examples(self, bin_dir: str):
        """
        Load pre-extracted bin examples from JSON files.
        
        Args:
            bin_dir: Directory containing bin_*_full.json files
        """
        print(f"\nLoading bin examples from: {bin_dir}")
        bin_dir_path = Path(bin_dir)
        
        # Load summary to get bin info
        summary_file = bin_dir_path / "bins_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Load each bin's full data
        for bin_info in summary['bins']:
            bin_id = bin_info['bin_id']
            bin_range = bin_info['bin_range']
            
            # Find the full data file
            full_file = bin_dir_path / f"bin_{bin_id}_{bin_range[0]}_{bin_range[1]}_full.json"
            
            with open(full_file, 'r') as f:
                bin_data = json.load(f)
            
            self.bin_examples[bin_id] = {
                'range': bin_range,
                'examples': bin_data['examples']
            }
            
            print(f"  ✓ Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: {len(bin_data['examples'])} examples")
        
        print(f"✓ Loaded {len(self.bin_examples)} bins\n")
    
    def build_user_turn(self, context: str, question: str,
                         include_instruction: bool = False) -> str:
        """
        Build a user turn with context and question.
        The task instruction lives in the system prompt; optionally it is also
        prepended to the first ICL user turn for extra emphasis.
        
        Args:
            context: The context paragraph
            question: The question to answer
            include_instruction: If True, prepend SYSTEM_PROMPT task instruction
            
        Returns:
            User turn text
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
        """
        Build an assistant turn with the answer.
        
        Args:
            answer: The answer text (or "NO_ANSWER")
            
        Returns:
            Assistant turn text
        """
        return f"ANSWER: {answer}"
    
    def calculate_max_k_for_bin(self, bin_id: int) -> int:
        """
        Calculate maximum k value for a bin given the token budget.

        The task instruction appears in the system prompt AND is prepended to the
        first ICL user turn (one-time fixed cost). Per-example cost is context
        tokens only (instruction is NOT repeated on turns 2..k).

        Budget equation:
            first_turn_instruction + k * avg_context <= token_budget - INSTRUCTION_TOKEN_OVERHEAD
            k <= (available_tokens - first_turn_instruction) / avg_context

        Args:
            bin_id: Bin ID (1-4)

        Returns:
            Maximum k that fits within budget
        """
        bin_range = self.bin_examples[bin_id]['range']
        avg_context_tokens = (bin_range[0] + bin_range[1]) / 2

        # Instruction is prepended to first ICL turn only — one-time fixed cost.
        first_turn_instruction_tokens = count_tokens(SYSTEM_PROMPT)
        available_tokens = self.token_budget - INSTRUCTION_TOKEN_OVERHEAD - first_turn_instruction_tokens
        max_k = int(available_tokens / avg_context_tokens)

        return max(1, max_k)  # At least k=1
    
    def sample_examples_from_bin(self, bin_id: int, k: int, seed: int = 42) -> List[Dict]:
        """
        Sample k random examples from a specific bin.
        
        Args:
            bin_id: Bin ID (1-4)
            k: Number of examples to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of sampled examples
        """
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
        
        Args:
            test_question: The test question
            test_context: The test context
            few_shot_examples: List of few-shot examples
            
        Returns:
            List of conversation turns in OpenAI format: [{"role": "user"/"assistant", "content": "..."}]
        """
        conversation = []
        
        # Add each few-shot example as a user-assistant turn pair
        for idx, example in enumerate(few_shot_examples):
            context = example.get('context_text', '')
            question = example.get('question_text', '')
            
            # Determine answer
            if example.get('is_impossible', False):
                answer = "NO_ANSWER"
            else:
                answer = example.get('answer_text', '').strip()
                if not answer:
                    answer = "NO_ANSWER"
            
            # User turn: Present context and question; repeat instruction on first turn only
            user_content = self.build_user_turn(context, question,
                                                include_instruction=(idx == 0))
            conversation.append({"role": "user", "content": user_content})
            
            # Assistant turn: Provide the answer for this example
            assistant_content = self.build_assistant_turn(answer)
            conversation.append({"role": "assistant", "content": assistant_content})
        
        # Final user turn: The actual test question
        test_user_content = self.build_user_turn(test_context, test_question)
        conversation.append({"role": "user", "content": test_user_content})
        
        return conversation
    
    def run_inference_batch(self, conversations: List[List[Dict[str, str]]], 
                           temperature: float = 0.0,
                           top_p: float = 1.0,
                           max_tokens: int = 128) -> List[str]:
        """
        Run batched inference on multiple multi-turn conversations.
        
        Args:
            conversations: List of conversations, each is a list of turns
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max tokens to generate
            
        Returns:
            List of generated answers
        """
        # Add system prompt to each conversation
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
            # Disable thinking mode for reasoning models (e.g., Qwen3 series)
            outputs = self.llm.chat(
                conversations_with_system, 
                sampling_params=sampling
                # chat_template_kwargs={"enable_thinking": False}
            )
            answers = [out.outputs[0].text.strip() for out in outputs]
            return answers
        except Exception as e:
            print(f"\n✗ Error during inference: {e}")
            return ["ERROR"] * len(conversations)
    
    def load_test_queries(self, test_file: str, max_queries: Optional[int] = None) -> List[Dict]:
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
                    
                    # Collect all answers
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
    
    def count_conversation_tokens(self, conversation: List[Dict[str, str]]) -> int:
        """
        Count total tokens in a multi-turn conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Total token count
        """
        total_tokens = 0
        for turn in conversation:
            total_tokens += count_tokens(turn['content'])
        return total_tokens
    
    def save_prompt_samples(self, conversations_data: List[Dict], n: int,
                             sample_dir: str, model_name: str) -> str:
        """
        Save the first n conversations as human-readable text for manual inspection.

        Args:
            conversations_data: List of conversation data dicts
            n: Number of samples to save
            sample_dir: Directory to write sample file
            model_name: Sanitised model name used in filename

        Returns:
            Path to the written sample file
        """
        os.makedirs(sample_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(sample_dir,
                                   f"{model_name}_prompt_samples_{timestamp}.txt")

        samples = conversations_data[:n]
        lines = []
        lines.append(f"Prompt samples — {model_name}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Showing first {len(samples)} conversations")
        lines.append(f"System prompt: {SYSTEM_PROMPT}")
        lines.append("")

        for idx, conv_data in enumerate(samples, 1):
            sep = "=" * 80
            lines.append(sep)
            lines.append(f"SAMPLE {idx}/{len(samples)}")
            lines.append(f"  Bin: {conv_data['bin_id']},  k: {conv_data['k']}")
            lines.append(f"  QA ID: {conv_data['qa_id']}")
            lines.append(f"  Conversation tokens (excl. system): {conv_data['conversation_tokens']}")
            lines.append(f"  Number of turns: {conv_data['num_turns']}")
            lines.append(sep)
            lines.append("")

            conversation = conv_data['conversation']
            # Label each turn
            icl_pair = 0
            i = 0
            while i < len(conversation):
                turn = conversation[i]
                role = turn['role'].upper()
                if i < len(conversation) - 1:  # ICL turns come in user/assistant pairs
                    if turn['role'] == 'user':
                        icl_pair += 1
                        lines.append(f"--- TURN {icl_pair} [USER — ICL EXAMPLE {icl_pair}] ---")
                    else:
                        lines.append(f"--- TURN {icl_pair} [ASSISTANT — ICL EXAMPLE {icl_pair}] ---")
                else:
                    lines.append("--- FINAL USER TURN [TEST QUERY] ---")
                lines.append(turn['content'])
                lines.append("")
                i += 1

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
                       output_dir: str = "length_bin_experiments_multiturn",
                       seed: int = 42,
                       num_runs: int = 5,
                       prompt_sample_dir: Optional[str] = None) -> List[str]:
        """
        Run the complete length bin experiments with multi-turn conversations.
        
        Args:
            test_queries_file: Path to Beat the AI dev JSON file
            bin_dir: Directory with extracted bin examples
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Max tokens to generate
            batch_size: Batch size for inference
            max_test_queries: Max number of test queries
            output_dir: Output directory
            seed: Base random seed
            num_runs: Number of runs with different seeds
            
        Returns:
            List of paths to output files (one per run)
        """
        print(f"\n{'='*70}")
        print(f"Length-Based Bin Experiments (MULTI-TURN) with Fixed Token Budget")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Token budget: {self.token_budget} tokens")
        print(f"Temperature: {temperature}")
        print(f"Batch size: {batch_size}")
        print(f"Base random seed: {seed}")
        print(f"Number of runs: {num_runs}")
        print(f"{'='*70}\n")
        
        # Load bin examples (once for all runs)
        self.load_bin_examples(bin_dir)
        
        # Load test queries (once for all runs)
        test_queries = self.load_test_queries(test_queries_file, max_test_queries)
        
        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.replace(':', '_').replace('/', '_')
        
        # Calculate max k for each bin
        first_turn_instruction_tokens = count_tokens(SYSTEM_PROMPT)
        print(f"Using fixed instruction overhead: {INSTRUCTION_TOKEN_OVERHEAD} tokens")
        print(f"Task instruction: in system prompt + first ICL user turn (one-time, {first_turn_instruction_tokens} tok)")
        print(f"Available tokens for ICL examples: {self.token_budget - INSTRUCTION_TOKEN_OVERHEAD - first_turn_instruction_tokens}\n")
        
        print(f"Calculating max k for each bin (per-example cost = context tokens only):")
        bin_k_ranges = {}
        for bin_id in sorted(self.bin_examples.keys()):
            max_k = self.calculate_max_k_for_bin(bin_id)
            bin_k_ranges[bin_id] = list(range(1, max_k + 1))
            bin_range = self.bin_examples[bin_id]['range']
            avg_context = (bin_range[0] + bin_range[1]) / 2
            print(f"  Bin {bin_id} [{bin_range[0]}-{bin_range[1]}]: ~{avg_context:.0f} tok/example → k=1 to k={max_k}")
        
        # Prepare all experiment configurations
        all_experiments = []
        for bin_id in sorted(self.bin_examples.keys()):
            for k in bin_k_ranges[bin_id]:
                all_experiments.append({
                    'bin_id': bin_id,
                    'k': k
                })
        
        print(f"\nTotal experiment configurations: {len(all_experiments)}")
        print(f"Total prompts per run: {len(all_experiments) * len(test_queries)}")
        print(f"Total prompts across {num_runs} runs: {len(all_experiments) * len(test_queries) * num_runs}\n")
        
        # Run experiments multiple times with different seeds
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
        
        # Print summary
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
        """Run a single experiment with a specific seed."""
        # Generate output filename
        if num_runs > 1:
            output_path = os.path.join(output_dir, 
                                       f"{model_name}_length_bin_budget_{self.token_budget}_multiturn_run_{run_idx + 1}_seed_{run_seed}_results.json")
        else:
            output_path = os.path.join(output_dir, 
                                       f"{model_name}_length_bin_budget_{self.token_budget}_multiturn_results.json")
        
        # Build all conversations
        print("Building all multi-turn conversations...")
        all_conversations_data = []
        
        for exp_config in tqdm(all_experiments, desc="Experiment configs"):
            bin_id = exp_config['bin_id']
            k = exp_config['k']
            
            for query_data in test_queries:
                # Sample examples from bin with current run seed
                few_shot_examples = self.sample_examples_from_bin(bin_id, k, run_seed)
                
                # Build multi-turn conversation
                conversation = self.build_multiturn_conversation(
                    query_data['question_text'],
                    query_data['context_text'],
                    few_shot_examples
                )
                
                # Count tokens in conversation
                conversation_tokens = self.count_conversation_tokens(conversation)
                
                # Calculate example token lengths (each example = user turn + assistant turn)
                example_token_lengths = []
                for i in range(0, len(conversation) - 1, 2):  # Skip last turn (test)
                    if i + 1 < len(conversation):
                        user_tokens = count_tokens(conversation[i]['content'])
                        assistant_tokens = count_tokens(conversation[i + 1]['content'])
                        example_token_lengths.append(user_tokens + assistant_tokens)
                
                all_conversations_data.append({
                    'bin_id': bin_id,
                    'k': k,
                    'qa_id': query_data['qa_id'],
                    'question': query_data['question_text'],
                    'is_impossible': query_data['is_impossible'],
                    'answers_text': query_data['answers_text'],
                    'conversation': conversation,
                    'conversation_tokens': conversation_tokens,
                    'example_token_lengths': example_token_lengths,
                    'num_turns': len(conversation)
                })
        
        print(f"✓ Built {len(all_conversations_data)} multi-turn conversations\n")

        # Save prompt samples for manual inspection (first run only)
        if prompt_sample_dir is not None:
            print("Saving prompt samples for manual inspection...")
            self.save_prompt_samples(all_conversations_data, 5, prompt_sample_dir, model_name)
            print("")

        # Run batched inference
        print(f"Running batched inference (batch_size={batch_size})...")
        all_answers = []
        
        for batch_data in tqdm(
            list(chunked(all_conversations_data, batch_size)),
            desc="Processing batches"
        ):
            batch_conversations = [d['conversation'] for d in batch_data]
            batch_answers = self.run_inference_batch(
                batch_conversations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            all_answers.extend(batch_answers)
        
        print(f"✓ Completed inference\n")
        
        # Organize results
        print("Organizing results...")
        results = {
            "metadata": {
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
        
        # Group by experiment configuration
        for exp_config in all_experiments:
            bin_id = exp_config['bin_id']
            k = exp_config['k']
            
            exp_results = {
                "bin_id": bin_id,
                "bin_range": self.bin_examples[bin_id]['range'],
                "k": k,
                "responses": []
            }
            
            # Find all results for this configuration
            for i, conv_data in enumerate(all_conversations_data):
                if conv_data['bin_id'] == bin_id and conv_data['k'] == k:
                    exp_results["responses"].append({
                        "qa_id": conv_data['qa_id'],
                        "question": conv_data['question'],
                        "is_impossible": conv_data['is_impossible'],
                        "answers_text": conv_data['answers_text'],
                        "conversation_tokens": conv_data['conversation_tokens'],
                        "example_token_lengths": conv_data['example_token_lengths'],
                        "num_turns": conv_data['num_turns'],
                        "llm_response": all_answers[i]
                    })
            
            results["experiments"].append(exp_results)
        
        # Write results
        print(f"Writing results to file...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Run {run_idx + 1}/{num_runs} complete!")
        print(f"  Seed: {run_seed}")
        print(f"  Results saved to: {output_path}")
        print(f"  Total conversations processed: {len(all_conversations_data)}\n")
        
        return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run length-based bin experiments with fixed token budget (MULTI-TURN version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (1000 token budget, 5 runs, multi-turn)
  python beat_the_ai_length_bin_experiments_multiturn.py \\
      --test-queries /path/to/dev-v2.0.json \\
      --bin-dir beat_the_ai_code/get_token_length_size/bin_examples
  
  # Use custom token budget
  python beat_the_ai_length_bin_experiments_multiturn.py \\
      --test-queries /path/to/dev-v2.0.json \\
      --bin-dir beat_the_ai_code/get_token_length_size/bin_examples \\
      --token-budget 1500
  
  # Test with fewer queries and single run
  python beat_the_ai_length_bin_experiments_multiturn.py \\
      --test-queries /path/to/dev-v2.0.json \\
      --bin-dir beat_the_ai_code/get_token_length_size/bin_examples \\
      --max-test-queries 100 \\
      --num-runs 1
        """
    )
    parser.add_argument("--test-queries", type=str, required=True,
                       help="Path to Beat the AI dev JSON file")
    parser.add_argument("--bin-dir", type=str, required=True,
                       help="Directory with extracted bin examples")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                       help="Model name or path (default: google/gemma-3-4b-it)")
    parser.add_argument("--token-budget", type=int, default=1000,
                       help="Token budget for context (default: 1000)")
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
    parser.add_argument("--output-dir", type=str, default="length_bin_experiments_multiturn",
                       help="Output directory (default: length_bin_experiments_multiturn)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed (default: 42)")
    parser.add_argument("--num-runs", type=int, default=5,
                       help="Number of runs with different seeds for robustness (default: 5)")
    parser.add_argument("--prompt-sample-dir", type=str, default=None,
                       help="If set, save the first 5 conversations as readable text to this directory (run 1 only)")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Run experiments
    runner = LengthBinExperimentRunnerMultiTurn(
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

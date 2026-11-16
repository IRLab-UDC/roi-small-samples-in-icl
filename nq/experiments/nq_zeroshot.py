#!/usr/bin/env python3
"""
NQ Zero-Shot Experiment

This script runs a ZERO-SHOT baseline on Natural Questions.
No ICL examples are provided — the prompt consists of the task instruction
(system prompt) followed by a single user turn with the test context and
question.

Key design:
- No example retrieval or selection
- Prompt = system prompt + user turn (CONTEXT + QUESTION)
- Multi-turn format kept for consistency: system → user → (LLM response)
- Single run per model (deterministic with temperature=0)

Output:
  <model>_nq_zeroshot_results.json

Fields per query:
  - example_id, question, llm_response
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from more_itertools import chunked

# System prompt — identical to the ICL experiments for fair comparison
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

NQ_DEV_PATH = "/mnt/gpu-fastdata/anxo/sigir26-short-QA-ICL/nq-dataset-format/dataset-dev-example/bin1-simplified-nq-dev.jsonl"


class NQZeroShotRunner:
    """
    Run LLM inference in zero-shot mode on Natural Questions.

    No ICL examples — prompt is system prompt + test context + test question.
    """

    def __init__(self, model: str = "google/gemma-3-4b-it",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: Optional[int] = None):
        print(f"✓ Initializing NQ Zero-Shot Runner...")
        print(f"  Model: {model}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")
        if max_model_len:
            print(f"  Max model length: {max_model_len} tokens")

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.model = model
        print(f"✓ NQ Zero-Shot Runner initialized")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_test_queries(self, test_file: str,
                          max_queries: Optional[int] = None) -> List[Dict]:
        """
        Load test queries from an NQ dev JSONL file.

        Each line is one example with question_text, document_text, example_id.
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

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_conversation(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        Build a zero-shot conversation: system prompt + single user turn.
        The user turn includes the task instruction for consistency with
        the ICL experiments (where the first user turn gets the instruction).
        """
        user_content = f"""CONTEXT:
<<<
{context}
>>>

QUESTION:
{question}"""
        return [{"role": "user", "content": user_content}]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference_batch(self, conversations: List[List[Dict[str, str]]],
                            temperature: float = 0.0,
                            top_p: float = 1.0,
                            max_tokens: int = 128) -> List[str]:
        """Run batched inference on multiple conversations."""
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

    def save_prompt_samples(self, test_queries: List[Dict],
                            n_queries: int,
                            sample_dir: str, model_name: str) -> str:
        """Save the first n_queries prompts as human-readable text."""
        os.makedirs(sample_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(
            sample_dir, f"{model_name}_nq_zeroshot_samples_{timestamp}.txt")

        samples = test_queries[:n_queries]
        lines = []
        lines.append(f"NQ Zero-Shot Prompt Samples — {model_name}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Showing {len(samples)} queries")
        lines.append(f"System prompt:\n{SYSTEM_PROMPT}")
        lines.append("")

        for q_idx, query_data in enumerate(samples, 1):
            sep = "=" * 80
            lines.append(sep)
            lines.append(f"QUERY {q_idx}/{len(samples)}  —  Example ID: {query_data['example_id']}")
            lines.append(sep)
            lines.append("")

            conversation = self.build_conversation(
                query_data['question_text'],
                query_data['context_text'],
            )
            lines.append("    [USER — TEST QUERY (zero-shot)]")
            lines.append(conversation[0]['content'])
            lines.append("")

        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"  ✓ Prompt samples saved to: {sample_path}")
        return sample_path

    # ------------------------------------------------------------------
    # Experiment orchestration
    # ------------------------------------------------------------------

    def run_experiments(self, test_queries_file: str,
                        temperature: float = 0.0,
                        top_p: float = 1.0,
                        max_tokens: int = 128,
                        batch_size: int = 8,
                        max_test_queries: Optional[int] = None,
                        output_dir: str = "nq_zeroshot",
                        prompt_sample_dir: Optional[str] = None) -> str:
        """Run the zero-shot experiment."""
        print(f"\n{'='*70}")
        print(f"NQ Zero-Shot Experiment")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"{'='*70}\n")

        test_queries = self.load_test_queries(test_queries_file, max_test_queries)
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.replace(':', '_').replace('/', '_')

        # Save prompt samples
        if prompt_sample_dir is not None:
            print("Saving prompt samples for manual inspection...")
            self.save_prompt_samples(test_queries, 5,
                                     prompt_sample_dir, model_name)
            print("")

        # Build conversations
        print("Building zero-shot conversations...")
        all_conv_data: List[Dict] = []
        for query_data in tqdm(test_queries, desc="Building conversations"):
            conversation = self.build_conversation(
                query_data['question_text'],
                query_data['context_text'],
            )
            all_conv_data.append({
                'example_id': query_data['example_id'],
                'question': query_data['question_text'],
                'conversation': conversation,
            })
        print(f"✓ Built {len(all_conv_data)} conversations\n")

        # Batched inference
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

        # Build results
        responses = []
        for i, conv_data in enumerate(all_conv_data):
            responses.append({
                'example_id': conv_data['example_id'],
                'question': conv_data['question'],
                'llm_response': all_answers[i],
            })

        results = {
            "metadata": {
                "dataset": "nq",
                "model": self.model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "total_queries": len(test_queries),
                "approach": "zero-shot",
                "timestamp": datetime.now().isoformat(),
            },
            "responses": responses,
        }

        output_path = os.path.join(
            output_dir, f"{model_name}_nq_zeroshot_results.json")
        print(f"Writing results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Zero-shot experiment complete!")
        print(f"  Queries: {len(test_queries)}  |  Output: {output_path}\n")
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NQ zero-shot experiment — no ICL examples, just task instruction + test query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python nq_zeroshot.py \\
      --test-queries /path/to/nq-dev.jsonl

  # Quick sanity check
  python nq_zeroshot.py \\
      --test-queries /path/to/nq-dev.jsonl \\
      --max-test-queries 20
        """
    )
    parser.add_argument("--test-queries", type=str, default=NQ_DEV_PATH,
                        help=f"Path to NQ dev JSONL file (default: {NQ_DEV_PATH})")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                        help="Single model name or path (ignored if --models is set)")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="One or more model names/paths to run sequentially")
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
    parser.add_argument("--output-dir", type=str,
                        default="NQ_code/get_token_length_size/zeroshot",
                        help="Output directory")
    parser.add_argument("--prompt-sample-dir", type=str, default=None,
                        help="If set, save the first 5 conversations here")

    args = parser.parse_args()

    models_to_run = args.models if args.models else [args.model]

    print(f"Models to run ({len(models_to_run)}): {models_to_run}")

    for model_idx, model_name in enumerate(models_to_run):
        print(f"\n{'#'*70}")
        print(f"MODEL {model_idx + 1}/{len(models_to_run)}: {model_name}")
        print(f"{'#'*70}\n")

        runner = NQZeroShotRunner(
            model=model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )

        runner.run_experiments(
            test_queries_file=args.test_queries,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_test_queries=args.max_test_queries,
            output_dir=args.output_dir,
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

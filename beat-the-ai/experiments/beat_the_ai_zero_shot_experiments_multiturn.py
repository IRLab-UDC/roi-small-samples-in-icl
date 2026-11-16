#!/usr/bin/env python3
"""
Beat the AI Zero-Shot Experiments

Runs zero-shot inference: each test query gets the task instruction (system prompt)
and the query, with NO in-context learning examples.
"""

import json
import os
import random
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from more_itertools import chunked

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import ES_HOST, ES_PORT, ES_INDEX_NAME, ES_SCHEME

from transformers import AutoTokenizer

# Tokenizer will be initialized with the model
tokenizer = None

# System prompt — contains full task instruction
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


class ZeroShotExperimentRunner:
    """Run zero-shot LLM inference experiments on Beat the AI."""

    def __init__(self, model: str = "google/gemma-3-4b-it",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90,
                 max_model_len: Optional[int] = None):
        print(f"✓ Initializing Zero-Shot Experiment Runner...")
        print(f"  Model: {model}")
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

    def build_user_turn(self, context: str, question: str) -> str:
        return f"""CONTEXT:
<<<
{context}
>>>

QUESTION:
{question}"""

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
                        print(f"✓ Loaded {len(test_queries)} test queries (limited)")
                        return test_queries

        print(f"✓ Loaded {len(test_queries)} test queries")
        return test_queries

    def run_inference_batch(self, conversations: List[List[Dict[str, str]]],
                           temperature: float = 0.0,
                           top_p: float = 1.0,
                           max_tokens: int = 128) -> List[str]:
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

    def run_experiments(self, test_queries_file: str,
                       temperature: float = 0.0,
                       top_p: float = 1.0,
                       max_tokens: int = 128,
                       batch_size: int = 60,
                       max_test_queries: Optional[int] = None,
                       output_dir: str = "zero_shot_experiments",
                       prompt_sample_dir: Optional[str] = None) -> str:
        print(f"\n{'='*70}")
        print(f"Zero-Shot Experiments")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Temperature: {temperature}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")

        test_queries = self.load_test_queries(test_queries_file, max_test_queries)

        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.replace(':', '_').replace('/', '_')

        # Build conversations (system + single user turn per query)
        print("Building zero-shot conversations...")
        conversations = []
        for q in tqdm(test_queries, desc="Queries"):
            user_content = self.build_user_turn(q['context_text'], q['question_text'])
            conversations.append([{"role": "user", "content": user_content}])

        print(f"✓ Built {len(conversations)} zero-shot conversations\n")

        # Save prompt samples
        if prompt_sample_dir:
            os.makedirs(prompt_sample_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_path = os.path.join(prompt_sample_dir,
                                       f"{model_name}_zero_shot_prompt_samples_{timestamp}.txt")
            lines = [
                f"Zero-shot prompt samples — {model_name}",
                f"Generated: {datetime.now().isoformat()}",
                f"System prompt:\n{SYSTEM_PROMPT}\n",
            ]
            for idx, conv in enumerate(conversations[:5], 1):
                lines.append("=" * 80)
                lines.append(f"SAMPLE {idx}/5")
                lines.append(f"  QA ID: {test_queries[idx-1]['qa_id']}")
                lines.append("=" * 80)
                lines.append(conv[0]['content'])
                lines.append("")
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            print(f"  ✓ Prompt samples saved to: {sample_path}\n")

        # Run batched inference
        print(f"Running batched inference (batch_size={batch_size})...")
        all_answers = []
        for batch_convs in tqdm(list(chunked(conversations, batch_size)),
                                desc="Processing batches"):
            batch_answers = self.run_inference_batch(
                batch_convs, temperature=temperature,
                top_p=top_p, max_tokens=max_tokens
            )
            all_answers.extend(batch_answers)
        print(f"✓ Completed inference\n")

        # Organize results
        output_path = os.path.join(output_dir,
                                   f"{model_name}_zero_shot_results.json")
        results = {
            "metadata": {
                "model": self.model,
                "approach": "zero-shot",
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "total_queries": len(test_queries),
                "timestamp": datetime.now().isoformat(),
            },
            "responses": []
        }

        for i, q in enumerate(test_queries):
            user_tokens = count_tokens(conversations[i][0]['content'])
            results["responses"].append({
                "qa_id": q['qa_id'],
                "question": q['question_text'],
                "is_impossible": q['is_impossible'],
                "answers_text": q['answers_text'],
                "user_tokens": user_tokens,
                "llm_response": all_answers[i],
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved to: {output_path}")
        print(f"  Total queries processed: {len(test_queries)}\n")
        return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run zero-shot experiments on Beat the AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-queries", type=str, required=True,
                       help="Path to Beat the AI dev JSON file")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it",
                       help="Model name or path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--max-test-queries", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="zero_shot_experiments")
    parser.add_argument("--prompt-sample-dir", type=str, default=None,
                       help="Save first 5 conversations as readable text")

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    runner = ZeroShotExperimentRunner(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    runner.run_experiments(
        test_queries_file=args.test_queries,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_test_queries=args.max_test_queries,
        output_dir=args.output_dir,
        prompt_sample_dir=args.prompt_sample_dir,
    )

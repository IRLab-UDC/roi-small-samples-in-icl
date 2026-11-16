"""
Build a comprehensive golden truth file from the NQ dev set.

For each example produces:
  - example_id
  - question_text
  - short_answers     : list of all distinct short-answer text spans (from all annotators)
  - long_answer       : long-answer text (first annotator with a valid long answer), or null
  - yes_no_answers    : list of YES/NO values from annotators that marked yes/no
  - yes_no_majority   : majority vote of yes_no_answers, or null
  - is_unanswerable   : True if NO annotator provided a short answer, long answer, or yes/no

Token offsets in the simplified NQ format refer to whitespace-split tokens
of the document_text field.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

DEV_JSONL_PATH = "/mnt/gpu-fastdata/anxo/sigir26-short-QA-ICL/nq-dataset-format/dataset-dev-example/simplified-nq-dev.jsonl"


def extract_text_from_tokens(document_text, start_token, end_token):
    """Extract text span from whitespace-split document tokens."""
    tokens = document_text.split(" ")
    return " ".join(tokens[start_token:end_token])


def process_example(example):
    """
    Process a single NQ dev example and extract all answer information.

    Returns a golden truth record.
    """
    doc_text = example["document_text"]
    annotations = example["annotations"]

    short_answers_set = []   # preserve order, deduplicate
    seen_short = set()
    long_answer_text = None
    yes_no_list = []

    for ann in annotations:
        # --- Short answers ---
        for sa in ann.get("short_answers", []):
            sa_text = extract_text_from_tokens(
                doc_text, sa["start_token"], sa["end_token"]
            )
            if sa_text and sa_text not in seen_short:
                seen_short.add(sa_text)
                short_answers_set.append(sa_text)

        # --- Long answer (keep first valid one) ---
        if long_answer_text is None:
            la = ann.get("long_answer", {})
            if la.get("candidate_index", -1) != -1:
                la_text = extract_text_from_tokens(
                    doc_text, la["start_token"], la["end_token"]
                )
                if la_text:
                    long_answer_text = la_text

        # --- Yes / No ---
        yn = ann.get("yes_no_answer", "NONE")
        if yn in ("YES", "NO"):
            yes_no_list.append(yn)

    # Majority vote for yes/no
    yes_no_majority = None
    if yes_no_list:
        counts = Counter(yes_no_list)
        yes_no_majority = counts.most_common(1)[0][0]

    # Unanswerable: no annotator gave any answer at all
    is_unanswerable = (
        len(short_answers_set) == 0
        and long_answer_text is None
        and len(yes_no_list) == 0
    )

    return {
        "example_id": example["example_id"],
        "question_text": example["question_text"],
        "short_answers": short_answers_set,
        "long_answer": long_answer_text,
        "yes_no_answers": yes_no_list,
        "yes_no_majority": yes_no_majority,
        "is_unanswerable": is_unanswerable,
    }


def build_golden_truth(input_path, output_path):
    """Read the dev JSONL and write the golden truth JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    stats = Counter()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing dev examples"):
            example = json.loads(line)
            record = process_example(example)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Track stats
            stats["total"] += 1
            if record["short_answers"]:
                stats["has_short_answer"] += 1
            if record["long_answer"] is not None:
                stats["has_long_answer"] += 1
            if record["yes_no_answers"]:
                stats["has_yes_no"] += 1
            if record["is_unanswerable"]:
                stats["unanswerable"] += 1

    print(f"\n✓ Golden truth saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Total examples:      {stats['total']}")
    print(f"  Has short answer:    {stats['has_short_answer']}")
    print(f"  Has long answer:     {stats['has_long_answer']}")
    print(f"  Has yes/no:          {stats['has_yes_no']}")
    print(f"  Unanswerable:        {stats['unanswerable']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build comprehensive golden truth file from NQ dev set"
    )
    parser.add_argument(
        "--input", type=str, default=DEV_JSONL_PATH,
        help=f"Path to simplified-nq-dev.jsonl (default: {DEV_JSONL_PATH})"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(DEV_JSONL_PATH).parent / "nq_dev_golden_truth.jsonl"),
        help="Output JSONL path (default: nq_dev_golden_truth.jsonl in same dir)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Build NQ Dev Golden Truth")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 60)

    build_golden_truth(args.input, args.output)

    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)

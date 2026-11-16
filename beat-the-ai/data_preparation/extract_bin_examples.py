#!/usr/bin/env python3
"""
Extract and save examples from Elasticsearch that belong to specific length bins.

This script:
1. Scrolls through Elasticsearch training examples
2. Computes token lengths using the same method as compute_length_distribution.py
3. Filters examples that fall into specified bins
4. Saves document IDs and metadata to JSON files for later retrieval
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import ES_HOST, ES_PORT, ES_INDEX_NAME, ES_SCHEME
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer

TOKENIZER_NAME = "google/gemma-3-4b-it"
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Use the exact format from build_prompts.py
PROMPT_TEMPLATE = """CONTEXT:
<<<
{context}
>>>

QUESTION:
{question}

ANSWER: {answer}"""

NO_ANSWER_STR = "NO_ANSWER"


def n_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(tok.encode(text, add_special_tokens=False))


def es_client():
    """Create Elasticsearch client."""
    return Elasticsearch([f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"], verify_certs=False, request_timeout=60)


def scroll_all_docs(es, source_fields, batch_size=1000, query=None, max_docs=None):
    """Scroll through all documents in Elasticsearch."""
    if query is None:
        query = {"match_all": {}}

    resp = es.search(
        index=ES_INDEX_NAME,
        size=batch_size,
        scroll="2m",
        _source=source_fields,
        query=query,
        track_total_hits=True,
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    yielded = 0
    while hits:
        for h in hits:
            yield h["_id"], h["_source"]  # Return both ID and source
            yielded += 1
            if max_docs is not None and yielded >= max_docs:
                es.clear_scroll(scroll_id=scroll_id)
                return

        resp = es.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    es.clear_scroll(scroll_id=scroll_id)


def extract_bin_examples(bin_ranges, max_docs=None, output_dir="bin_examples"):
    """
    Extract examples from Elasticsearch that fall into specified bins.
    
    Args:
        bin_ranges: List of tuples [(min_length, max_length), ...]
        max_docs: Maximum number of documents to process
        output_dir: Directory to save bin example files
        
    Returns:
        Dictionary mapping bin_id to list of example data
    """
    es = es_client()
    fields = ["question_text", "context_text", "answer_text", "is_impossible"]
    
    # Initialize storage for each bin
    bin_examples = {i: [] for i in range(len(bin_ranges))}
    bin_counts = {i: 0 for i in range(len(bin_ranges))}
    
    print(f"\n{'='*60}")
    print("Extracting Examples for Specified Bins")
    print(f"{'='*60}")
    for i, (min_len, max_len) in enumerate(bin_ranges, 1):
        print(f"Bin {i}: [{min_len:3d}, {max_len:3d}] tokens")
    print(f"{'='*60}\n")
    
    total_processed = 0
    total_matched = 0
    
    # Process all documents
    for doc_id, doc in tqdm(scroll_all_docs(es, fields, batch_size=1000, max_docs=max_docs),
                            desc="Processing documents", total=max_docs):
        total_processed += 1
        
        # Extract fields
        q = doc.get("question_text", "") or ""
        c = doc.get("context_text", "") or ""
        impossible = bool(doc.get("is_impossible", False))
        a = NO_ANSWER_STR if impossible else (doc.get("answer_text", "") or "")
        
        # Compute token length
        demo = PROMPT_TEMPLATE.format(context=c, question=q, answer=a)
        token_length = n_tokens(demo)
        
        # Check which bin this example belongs to
        for bin_id, (min_len, max_len) in enumerate(bin_ranges):
            if min_len <= token_length <= max_len:
                bin_examples[bin_id].append({
                    "doc_id": doc_id,
                    "token_length": token_length,
                    "question_text": q,
                    "context_text": c,
                    "answer_text": a,
                    "is_impossible": impossible
                })
                bin_counts[bin_id] += 1
                total_matched += 1
                break  # Each example belongs to at most one bin
    
    print(f"\n{'='*60}")
    print("Extraction Summary")
    print(f"{'='*60}")
    print(f"Total documents processed: {total_processed:,}")
    print(f"Total examples matched:    {total_matched:,}")
    print(f"\nExamples per bin:")
    for bin_id, (min_len, max_len) in enumerate(bin_ranges):
        count = bin_counts[bin_id]
        print(f"  Bin {bin_id + 1} [{min_len:3d}, {max_len:3d}]: {count:6,} examples")
    print(f"{'='*60}\n")
    
    return bin_examples


def save_bin_examples(bin_examples, bin_ranges, output_dir="bin_examples"):
    """
    Save bin examples to JSON files.
    
    Args:
        bin_examples: Dictionary mapping bin_id to list of example data
        bin_ranges: List of tuples [(min_length, max_length), ...]
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save each bin to a separate file
    for bin_id, examples in bin_examples.items():
        min_len, max_len = bin_ranges[bin_id]
        
        # Full data file (includes all fields for verification)
        full_file = output_path / f"bin_{bin_id + 1}_{min_len}_{max_len}_full.json"
        with open(full_file, 'w') as f:
            json.dump({
                "bin_id": bin_id + 1,
                "bin_range": [min_len, max_len],
                "num_examples": len(examples),
                "examples": examples
            }, f, indent=2)
        
        # IDs only file (lightweight, for quick loading)
        ids_file = output_path / f"bin_{bin_id + 1}_{min_len}_{max_len}_ids.json"
        doc_ids = [ex["doc_id"] for ex in examples]
        with open(ids_file, 'w') as f:
            json.dump({
                "bin_id": bin_id + 1,
                "bin_range": [min_len, max_len],
                "num_examples": len(doc_ids),
                "doc_ids": doc_ids
            }, f, indent=2)
        
        print(f"✓ Bin {bin_id + 1} saved:")
        print(f"  Full data: {full_file} ({full_file.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"  IDs only:  {ids_file} ({ids_file.stat().st_size / 1024:.2f} KB)")
    
    # Save summary file
    summary_file = output_path / "bins_summary.json"
    summary = {
        "total_bins": len(bin_ranges),
        "bins": [
            {
                "bin_id": i + 1,
                "bin_range": [min_len, max_len],
                "num_examples": len(bin_examples[i])
            }
            for i, (min_len, max_len) in enumerate(bin_ranges)
        ],
        "total_examples": sum(len(examples) for examples in bin_examples.values()),
        "files": {
            "full_data": [f"bin_{i + 1}_{min_len}_{max_len}_full.json" 
                         for i, (min_len, max_len) in enumerate(bin_ranges)],
            "ids_only": [f"bin_{i + 1}_{min_len}_{max_len}_ids.json" 
                        for i, (min_len, max_len) in enumerate(bin_ranges)]
        }
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"\n{'='*60}")
    print("✓ All bin examples saved successfully!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract examples from Elasticsearch that belong to specific length bins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 4 custom bins (your experiment setup)
  python extract_bin_examples.py --bins "48,84" "195,231" "342,377" "452,487"
  
  # Process all documents (default)
  python extract_bin_examples.py --bins "48,84" "195,231" "342,377" "452,487" --max-docs 130000
  
  # Quick test with fewer documents
  python extract_bin_examples.py --bins "48,84" "195,231" --max-docs 5000
  
  # Custom output directory
  python extract_bin_examples.py --bins "48,84" "195,231" "342,377" "452,487" --output my_bins
        """
    )
    parser.add_argument(
        "--bins",
        type=str,
        nargs='+',
        required=True,
        help='Bin ranges as "min,max" strings. Example: "48,84" "195,231" "342,377" "452,487"'
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all documents)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bin_examples",
        help="Output directory for bin example files (default: bin_examples)"
    )
    args = parser.parse_args()
    
    # Parse bin ranges
    try:
        bin_ranges = []
        for bin_str in args.bins:
            min_len, max_len = map(int, bin_str.split(','))
            bin_ranges.append((min_len, max_len))
    except ValueError:
        print("✗ Error: Invalid bin format. Use 'min,max' format (e.g., '48,84')")
        return
    
    print("=" * 60)
    print("Extract Bin Examples from Elasticsearch")
    print("=" * 60)
    print(f"Tokenizer:      {TOKENIZER_NAME}")
    print(f"Max documents:  {args.max_docs if args.max_docs else 'All'}")
    print(f"Output dir:     {args.output}")
    print(f"Number of bins: {len(bin_ranges)}")
    print("=" * 60)
    
    # Extract examples
    bin_examples = extract_bin_examples(
        bin_ranges=bin_ranges,
        max_docs=args.max_docs,
        output_dir=args.output
    )
    
    # Save to files
    save_bin_examples(bin_examples, bin_ranges, output_dir=args.output)
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review the bin_examples/*.json files")
    print("2. Use the *_ids.json files for quick loading of document IDs")
    print("3. Use the *_full.json files for verification and analysis")
    print("4. Retrieve random subsets from Elasticsearch using the saved IDs")
    print("=" * 60)


if __name__ == "__main__":
    main()

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch

from transformers import AutoTokenizer
import sys
sys.path.append(str(Path(__file__).parent.parent))
# Elasticsearch Configuration
ES_HOST = "10.56.35.149"
ES_PORT = 9208
ES_INDEX_NAME = "anxo_nq_training_samples"
ES_SCHEME = "http"  # Change to "https" if using SSL

TOKENIZER_NAME = "google/gemma-3-4b-it"  # pick the tokenizer matching your eval model family
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Use the exact format from nq_build_prompts.py for accurate token counting
PROMPT_TEMPLATE = """CONTEXT:
<<<
{context}
>>>

QUESTION:
{question}

ANSWER: {answer}"""

NO_ANSWER_STR = "NO_ANSWER"

def n_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def es_client():
    return Elasticsearch([f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"], verify_certs=False, request_timeout=60)

def scroll_all_docs(es, source_fields, batch_size=1000, query=None, max_docs=None):
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
            yield h["_source"]
            yielded += 1
            if max_docs is not None and yielded >= max_docs:
                es.clear_scroll(scroll_id=scroll_id)
                return

        resp = es.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    es.clear_scroll(scroll_id=scroll_id)

def compute_demo_lengths(max_docs=310000):
    """
    Compute token lengths for all NQ training examples in Elasticsearch.
    
    Args:
        max_docs: Maximum number of documents to process (None for all)
        
    Returns:
        numpy array of token lengths
    """
    es = es_client()
    fields = ["question_text", "document_text", "short_answer", "long_answer",
              "has_short_answer", "has_long_answer"]

    lengths = []
    for doc in tqdm(scroll_all_docs(es, fields, batch_size=1000, max_docs=max_docs),
                    desc="Computing demo lengths", total=max_docs):
        q = doc.get("question_text", "") or ""
        c = doc.get("document_text", "") or ""

        # Use short answer if available, otherwise long answer, otherwise NO_ANSWER
        has_short = bool(doc.get("has_short_answer", False))
        has_long = bool(doc.get("has_long_answer", False))

        if has_short:
            a = (doc.get("short_answer", "") or "").strip()
        elif has_long:
            a = (doc.get("long_answer", "") or "").strip()
        else:
            a = NO_ANSWER_STR

        if not a:
            a = NO_ANSWER_STR

        demo = PROMPT_TEMPLATE.format(context=c, question=q, answer=a)
        lengths.append(n_tokens(demo))

    lengths = np.array(lengths, dtype=np.int32)
    print(f"\n✓ Computed lengths for {len(lengths)} documents")
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {np.mean(lengths):.1f} tokens")
    print(f"  Median: {np.median(lengths):.1f} tokens")
    print(f"  Std:    {np.std(lengths):.1f} tokens")
    print(f"  Min:    {np.min(lengths)} tokens")
    print(f"  Max:    {np.max(lengths)} tokens")
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}: {int(np.percentile(lengths, p)):4d} tokens")
    
    return lengths


def calculate_optimal_bins_freedman_diaconis(lengths):
    """
    Calculate optimal number of bins using Freedman-Diaconis rule.
    
    Bin width = 2 * IQR / n^(1/3)
    Number of bins = (max - min) / bin_width
    
    Args:
        lengths: Array of token lengths
        
    Returns:
        Optimal number of bins (int)
    """
    n = len(lengths)
    q75, q25 = np.percentile(lengths, [75, 25])
    iqr = q75 - q25
    
    if iqr == 0:
        return int(np.ceil(np.log2(n) + 1))
    
    bin_width = 2 * iqr / np.cbrt(n)
    data_range = np.max(lengths) - np.min(lengths)
    num_bins = int(np.ceil(data_range / bin_width))
    num_bins = max(3, min(num_bins, 20))
    
    return num_bins


def analyze_length_bins_equal_count(lengths, num_bins=3):
    """
    Analyze length bins with equal number of examples per bin (quantile-based).
    """
    sorted_indices = np.argsort(lengths)
    sorted_lengths = lengths[sorted_indices]
    
    total_samples = len(sorted_lengths)
    samples_per_bin = total_samples // num_bins
    
    bins_data = []
    for i in range(num_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < num_bins - 1 else total_samples
        
        bin_lengths = sorted_lengths[start_idx:end_idx]
        bin_min = int(bin_lengths[0])
        bin_max = int(bin_lengths[-1])
        bin_samples = len(bin_lengths)
        bin_mean = float(np.mean(bin_lengths))
        bin_median = float(np.median(bin_lengths))
        
        bins_data.append({
            "bin_id": i + 1,
            "min_length": bin_min,
            "max_length": bin_max,
            "num_samples": bin_samples,
            "mean_length": bin_mean,
            "median_length": bin_median,
            "length_distribution": bin_lengths.tolist()
        })
        
        print(f"Bin {i+1}: [{bin_min:4d}, {bin_max:4d}] tokens  "
              f"({bin_samples:6d} samples, mean={bin_mean:.1f}, median={bin_median:.1f})")
    
    return bins_data


def analyze_length_bins_equal_width(lengths, num_bins=3):
    """
    Analyze length bins with equal width (better differentiation in token length ranges).
    """
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    
    bin_edges = np.linspace(min_length, max_length, num_bins + 1)
    
    bins_data = []
    for i in range(num_bins):
        bin_min = bin_edges[i]
        bin_max = bin_edges[i + 1]
        
        if i == num_bins - 1:
            mask = (lengths >= bin_min) & (lengths <= bin_max)
        else:
            mask = (lengths >= bin_min) & (lengths < bin_max)
        
        bin_lengths = lengths[mask]
        
        if len(bin_lengths) == 0:
            continue
        
        bin_samples = len(bin_lengths)
        bin_mean = float(np.mean(bin_lengths))
        bin_median = float(np.median(bin_lengths))
        
        bins_data.append({
            "bin_id": i + 1,
            "min_length": int(np.ceil(bin_min)),
            "max_length": int(np.floor(bin_max)),
            "num_samples": bin_samples,
            "mean_length": bin_mean,
            "median_length": bin_median,
            "length_distribution": sorted(bin_lengths.tolist())
        })
        
        print(f"Bin {i+1}: [{int(np.ceil(bin_min)):4d}, {int(np.floor(bin_max)):4d}] tokens  "
              f"({bin_samples:6d} samples, mean={bin_mean:.1f}, median={bin_median:.1f})")
    
    return bins_data


def analyze_length_bins_freedman_diaconis(lengths):
    """
    Analyze length bins using Freedman-Diaconis rule for optimal bin width.
    """
    num_bins = calculate_optimal_bins_freedman_diaconis(lengths)
    print(f"Freedman-Diaconis rule suggests {num_bins} bins")
    return analyze_length_bins_equal_width(lengths, num_bins=num_bins)


def analyze_multiple_bin_configurations(lengths, min_bins=2, max_bins=10, method="equal_count"):
    """
    Analyze multiple bin configurations from min_bins to max_bins.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Bin Configurations ({min_bins} to {max_bins} bins)")
    print(f"Method: {method}")
    print(f"{'='*60}\n")
    
    all_configs = {}
    
    if method == "freedman_diaconis":
        print(f"\nFreedman-Diaconis Configuration (auto-calculated):")
        print(f"{'-'*60}")
        bins_data = analyze_length_bins_freedman_diaconis(lengths)
        num_bins_fd = len(bins_data)
        all_configs[f"{num_bins_fd}_bins_fd"] = bins_data
    else:
        for num_bins in range(min_bins, max_bins + 1):
            print(f"\n{num_bins}-Bin Configuration:")
            print(f"{'-'*60}")
            
            if method == "equal_count":
                bins_data = analyze_length_bins_equal_count(lengths, num_bins=num_bins)
            elif method == "equal_width":
                bins_data = analyze_length_bins_equal_width(lengths, num_bins=num_bins)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'equal_count', 'equal_width', or 'freedman_diaconis'")
            
            all_configs[f"{num_bins}_bins"] = bins_data
    
    return all_configs


def save_distribution_data(lengths, bin_configurations, output_path="length_distribution.json"):
    """
    Save the length distribution data to JSON for later use.
    """
    data = {
        "total_examples": int(len(lengths)),
        "statistics": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths))
        },
        "percentiles": {
            f"p{p}": int(np.percentile(lengths, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        "bin_configurations": bin_configurations
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Distribution data saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Contains bin configurations from {min([int(k.split('_')[0]) for k in bin_configurations.keys()])} to {max([int(k.split('_')[0]) for k in bin_configurations.keys()])} bins")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute and analyze token length distribution of NQ training examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: equal count bins (quantile-based, equal samples per bin)
  python compute_length_distribution.py --max-docs 310000 --min-bins 2 --max-bins 10
  
  # Equal width bins (better length differentiation, varying sample counts)
  python compute_length_distribution.py --max-docs 310000 --min-bins 2 --max-bins 10 --method equal_width
  
  # Freedman-Diaconis rule (auto-calculates optimal number of bins)
  python compute_length_distribution.py --max-docs 310000 --method freedman_diaconis
  
  # Quick test with fewer documents
  python compute_length_distribution.py --max-docs 5000 --method equal_width
        """
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=310000,
        help="Maximum number of documents to process (default: 310000, ~all NQ train)"
    )
    parser.add_argument(
        "--min-bins",
        type=int,
        default=2,
        help="Minimum number of bins to analyze (default: 2, ignored for freedman_diaconis)"
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=10,
        help="Maximum number of bins to analyze (default: 10, ignored for freedman_diaconis)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="equal_count",
        choices=["equal_count", "equal_width", "freedman_diaconis"],
        help="Binning method: 'equal_count' (equal samples per bin), "
             "'equal_width' (equal token length ranges), "
             "'freedman_diaconis' (auto-calculate optimal bins) (default: equal_count)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="length_distribution.json",
        help="Output path for distribution data (default: length_distribution.json)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("NQ Token Length Distribution Analysis")
    print("=" * 60)
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print(f"Max documents: {args.max_docs}")
    if args.method == "freedman_diaconis":
        print(f"Binning method: {args.method} (auto-calculates bins)")
    else:
        print(f"Bin range: {args.min_bins} to {args.max_bins} bins")
        print(f"Binning method: {args.method}")
    print("=" * 60)
    
    # Compute lengths
    lengths = compute_demo_lengths(max_docs=args.max_docs)
    
    # Analyze multiple bin configurations
    bin_configurations = analyze_multiple_bin_configurations(
        lengths, 
        min_bins=args.min_bins, 
        max_bins=args.max_bins,
        method=args.method
    )
    
    # Save data with all bin configurations
    save_distribution_data(lengths, bin_configurations, output_path=args.output)
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print(f"✓ You can now use this data to create plots with matplotlib")
    print("=" * 60)

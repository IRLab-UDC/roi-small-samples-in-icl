#!/usr/bin/env python3
"""
Plot F1 vs token budget for length-bin experiments.

Usage:
    python plot_length_bin_results.py data_gemma4b.json
    python plot_length_bin_results.py data_gemma4b.json --output gemma4b.pdf
    python plot_length_bin_results.py data_gemma4b_multiturn.json \
        --random data_gemma4b_random_multiturn.json --output gemma4b_with_random.pdf
    python plot_length_bin_results.py data_gemma4b_multiturn.json --legend
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINESTYLES = {"Bin 1": "-",  "Bin 2": "--", "Bin 3": "-.", "Bin 4": ":"}
MARKERS    = {"Bin 1": "o",  "Bin 2": "s",  "Bin 3": "^",  "Bin 4": "D"}

TICK_FONTSIZE  = 18
LABEL_FONTSIZE = 20


def main():
    parser = argparse.ArgumentParser(
        description="Plot F1 vs token budget from a JSON results file"
    )
    parser.add_argument("data", help="Path to JSON data file")
    parser.add_argument("--random", metavar="RANDOM_JSON",
                        help="Optional random-baseline JSON file to overlay as an extra line")
    parser.add_argument("--output", "-o",
                        help="Save plot to this file instead of showing interactively")
    parser.add_argument("--max-tokens", type=float, default=None,
                        help="Only show data points up to this many tokens on the x-axis")
    parser.add_argument("--legend", action="store_true", default=False,
                        help="Show legend on the plot")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    xlabel    = data["xlabel"]
    avg_len   = data["avg_len"]
    bins_data = data["bins"]

    YMIN = 48
    YMAX = 69
    YSTEP = 5

    fig, ax = plt.subplots(figsize=(10, 3.5))
    x_all_points = []

    for name, values in bins_data.items():
        if name == "Zero-shot":
            continue
        y = np.array(values, dtype=float)
        k = np.arange(1, len(y) + 1)
        x = k * avg_len[name]
        mask = ~np.isnan(y)
        if args.max_tokens is not None:
            mask = mask & (x <= args.max_tokens)
        ax.plot(
            x[mask], y[mask],
            linestyle=LINESTYLES[name],
            marker=MARKERS[name],
            linewidth=1.5,
            markersize=4,
            label=f"{name} (avg={avg_len[name]:.1f})"
        )
        x_all_points.extend(x[mask].tolist())

    # --- optional random baseline overlay ---
    if args.random:
        with open(args.random) as f:
            rdata = json.load(f)
        ks_dict = rdata["ks"]
        sorted_ks = sorted(ks_dict.items(), key=lambda kv: int(kv[0]))
        rx = np.array([v["avg_tokens"] for _, v in sorted_ks])
        ry = np.array([v["avg_f1"]     for _, v in sorted_ks])
        if args.max_tokens is not None:
            rmask = rx <= args.max_tokens
            rx, ry = rx[rmask], ry[rmask]
        ax.plot(rx, ry, linestyle="-", marker="x", linewidth=1.5, markersize=5,
                color="black", label="Random baseline")
        x_all_points.extend(rx.tolist())

    ax.set_xlabel("Token budget", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("F1", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    if args.legend:
        ax.legend(ncol=2, frameon=True, fontsize=TICK_FONTSIZE - 1)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(np.arange(YMIN, YMAX + 1, YSTEP))
    ax.set_xlim(0, 4000)
    ax.set_xticks(np.arange(0, 4001, 500))

    fig.tight_layout(pad=0.1)
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.01)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

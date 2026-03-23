#!/usr/bin/env python3
"""Generate scaling comparison figures from benchmark_results.csv.

Produces:
    figures/scaling_runtime.png   -- runtime vs #pairs (log-log)
    figures/scaling_speedup.png   -- speedup vs #pairs
    figures/scaling_combined.png  -- 2-panel summary

Usage:
    python plot_scaling.py
    python plot_scaling.py --csv benchmark_results.csv --out-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EXPERIMENT_DIR = Path(__file__).resolve().parent

PAIR_COLOR = "#1f77b4"
NODE_COLOR = "#2ca02c"
GREY = "#7f7f7f"

MARKERS = {50: "o", 100: "s", 200: "D"}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 180,
    })


def load_csv(path: Path):
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "n_samples": int(r["n_samples"]),
                "n_pairs": int(r["n_pairs"]),
                "pair_ms": float(r["pair_ms"]),
                "pair_std": float(r["pair_std"]),
                "node_ms": float(r["node_ms"]),
                "node_std": float(r["node_std"]),
                "speedup": float(r["speedup"]),
            })
    return rows


def plot_runtime(rows, out_path: Path):
    """Log-log runtime vs number of pairs."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_samples"], []).append(r)

    for n_samples, data in sorted(by_n.items()):
        pairs = [d["n_pairs"] for d in data]
        pair_ms = [d["pair_ms"] for d in data]
        node_ms = [d["node_ms"] for d in data]
        m = MARKERS.get(n_samples, "^")

        ax.plot(pairs, pair_ms, f"-{m}", color=PAIR_COLOR, markersize=5, lw=1.5,
                label=f"Pairwise (n={n_samples})", alpha=0.8)
        ax.plot(pairs, node_ms, f"--{m}", color=NODE_COLOR, markersize=5, lw=1.5,
                label=f"Node-time (n={n_samples})", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of pairs")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Runtime scaling: Pairwise vs Node-time model")

    ax.axhline(1000, color=GREY, ls=":", lw=0.7, alpha=0.5)
    ax.axhline(60_000, color=GREY, ls=":", lw=0.7, alpha=0.5)
    ax.text(1.5, 1100, "1 s", fontsize=8, color=GREY)
    ax.text(1.5, 66_000, "1 min", fontsize=8, color=GREY)

    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_speedup(rows, out_path: Path):
    """Speedup vs number of pairs."""
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_samples"], []).append(r)

    for n_samples, data in sorted(by_n.items()):
        pairs = [d["n_pairs"] for d in data]
        speedup = [d["speedup"] for d in data]
        m = MARKERS.get(n_samples, "^")
        ax.plot(pairs, speedup, f"-{m}", markersize=5, lw=1.5,
                label=f"n={n_samples}", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of pairs")
    ax.set_ylabel("Speedup (pairwise / node-time)")
    ax.set_title("Speedup: Node-time model vs Pairwise model")

    ax.axhline(1, color=GREY, ls="--", lw=0.8)
    ax.axhline(10, color=GREY, ls=":", lw=0.5, alpha=0.5)
    ax.axhline(100, color=GREY, ls=":", lw=0.5, alpha=0.5)

    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_combined(rows, out_path: Path):
    """Two-panel figure: runtime + speedup side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_samples"], []).append(r)

    for n_samples, data in sorted(by_n.items()):
        pairs = [d["n_pairs"] for d in data]
        pair_ms = [d["pair_ms"] for d in data]
        node_ms = [d["node_ms"] for d in data]
        speedup = [d["speedup"] for d in data]
        m = MARKERS.get(n_samples, "^")

        ax1.plot(pairs, pair_ms, f"-{m}", color=PAIR_COLOR, markersize=5, lw=1.5,
                 label=f"Pairwise (n={n_samples})", alpha=0.8)
        ax1.plot(pairs, node_ms, f"--{m}", color=NODE_COLOR, markersize=5, lw=1.5,
                 label=f"Node-time (n={n_samples})", alpha=0.8)
        ax2.plot(pairs, speedup, f"-{m}", markersize=5, lw=1.5,
                 label=f"n={n_samples}", alpha=0.8)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of pairs")
    ax1.set_ylabel("Inference time (ms)")
    ax1.set_title("A) Runtime scaling")
    ax1.axhline(1000, color=GREY, ls=":", lw=0.7, alpha=0.5)
    ax1.axhline(60_000, color=GREY, ls=":", lw=0.7, alpha=0.5)
    ax1.text(1.5, 1100, "1 s", fontsize=8, color=GREY)
    ax1.text(1.5, 66_000, "1 min", fontsize=8, color=GREY)
    ax1.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=8)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of pairs")
    ax2.set_ylabel("Speedup (pairwise / node-time)")
    ax2.set_title("B) Speedup")
    ax2.axhline(1, color=GREY, ls="--", lw=0.8)
    ax2.axhline(10, color=GREY, ls=":", lw=0.5, alpha=0.5)
    ax2.axhline(100, color=GREY, ls=":", lw=0.5, alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Pairwise O(n²) vs Node-time O(1) inference scaling", fontsize=13, y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Plot scaling benchmark results")
    ap.add_argument("--csv", type=str,
                    default=str(EXPERIMENT_DIR / "benchmark_results.csv"))
    ap.add_argument("--out-dir", type=str,
                    default=str(EXPERIMENT_DIR / "figures"))
    args = ap.parse_args()

    _setup_style()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run benchmark_scaling.py first.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} benchmark rows from {csv_path}")

    plot_runtime(rows, out_dir / "scaling_runtime.png")
    plot_speedup(rows, out_dir / "scaling_speedup.png")
    plot_combined(rows, out_dir / "scaling_combined.png")

    print(f"\nDone — figures saved to {out_dir}/")


if __name__ == "__main__":
    main()

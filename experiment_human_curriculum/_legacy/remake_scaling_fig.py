#!/usr/bin/env python3
"""Regenerate scaling figure (paper Figure 4) from benchmark CSV with larger fonts."""
import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "experiment" / "paper_figures" / "table2_benchmark.csv"
OUT = REPO / "experiment" / "paper_figures" / "fig5_scaling.png"

BLUE = "#2271b2"
GREEN = "#228833"
ORANGE = "#ee6100"
GREY = "#666666"

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
})

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if int(r["n"]) >= 30]

ns = [int(r["n"]) for r in rows]
t_pw = [float(r["pairwise"]) for r in rows]
t_nt = [float(r["node_time"]) for r in rows]
t_ts = [float(r["tsinfer"]) for r in rows]
t_fp = [float(r["full_pipeline"]) for r in rows]
speedups_nt = [float(r["pairwise"]) / float(r["node_time"]) for r in rows]
speedups_fp = [float(r["pairwise"]) / float(r["full_pipeline"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(ns, t_pw, "o-", color=BLUE, lw=2.0, markersize=8, label="Pairwise Mamba")
ax1.plot(ns, t_nt, "s-", color=GREEN, lw=2.0, markersize=8, label="Node-time only")
ax1.plot(ns, t_ts, "^-", color=GREY, lw=1.5, markersize=7, label="tsinfer", alpha=0.8)
ax1.plot(ns, t_fp, "D--", color=ORANGE, lw=1.5, markersize=7, label="Full pipeline", alpha=0.9)

ax1.set_xlabel("Number of haplotypes ($n$)")
ax1.set_ylabel("Inference time (s)")
ax1.set_title("a", fontweight="bold", loc="left")
ax1.legend(loc="upper left", framealpha=0.9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

x = np.arange(len(ns))
w = 0.35
bars1 = ax2.bar(x - w / 2, speedups_nt, w, color=GREEN, alpha=0.85,
                edgecolor="white", linewidth=0.5, label="Node-time only")
bars2 = ax2.bar(x + w / 2, speedups_fp, w, color=ORANGE, alpha=0.85,
                edgecolor="white", linewidth=0.5, label="Full pipeline")

for bar, s in zip(bars1, speedups_nt):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{s:.0f}$\\times$", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color=GREEN)
for bar, s in zip(bars2, speedups_fp):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{s:.0f}$\\times$", ha="center", va="bottom", fontsize=10,
             fontweight="bold", color=ORANGE)

ax2.set_xticks(x)
ax2.set_xticklabels([str(n) for n in ns])
ax2.set_xlabel("Number of haplotypes ($n$)")
ax2.set_ylabel("Speedup over pairwise Mamba")
ax2.set_title("b", fontweight="bold", loc="left")
ax2.legend(loc="upper left", framealpha=0.9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(w_pad=2.0)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT} ({fig.get_size_inches()[0]*300:.0f}x{fig.get_size_inches()[1]*300:.0f})")

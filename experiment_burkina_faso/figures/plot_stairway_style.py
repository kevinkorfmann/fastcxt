#!/usr/bin/env python3
"""Plot Burkina Faso IICR in Stairway-Plot style (absolute time in ka, Ne in millions).

Uses the same scaling assumptions as Miles et al. (2017) Fig. 9a:
  - mu = 3.5e-9 per bp per generation (midpoint of 2.8e-9 to 5.5e-9)
  - generation time = 1/11 years (11 generations per year)

Plots inversion-free arms (3L, 3R) for the cleanest demographic signal.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Parameters matching Miles et al. 2017 caption
# ---------------------------------------------------------------------------
MU_LOW = 2.8e-9      # per bp per generation
MU_HIGH = 5.5e-9
MU_MID = 3.5e-9       # geometric midpoint ~ 3.9e-9, but 3.5e-9 is commonly used
GEN_PER_YEAR = 11

RESULTS_DIR = Path("/tmp/het_results")
EDGE_TRIM_MB = 2.0
DEMOG_MIN_ACC = 0.5

# ---------------------------------------------------------------------------
# Data loading (copied from generate_all_figures.py)
# ---------------------------------------------------------------------------

def load_group(pop, arm, group, pair_type):
    d = RESULTS_DIR / pop / arm / f"{group}_{pair_type}"
    if not (d / "means.npz").exists():
        return None
    means = np.load(d / "means.npz")["means"]
    index_map = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    config = json.load(open(d / "config.json"))
    return dict(means=means, index_map=index_map, blocks=blocks, config=config)


def _get_masked_tmrcas(data, arm):
    means = data["means"]
    idx = data["index_map"]
    blocks = data["blocks"]
    n_blocks = len(blocks)
    n_pairs = int(idx[:, 1].max()) + 1
    n_win = means.shape[1]

    arr = np.full((n_blocks, n_pairs, n_win), np.nan)
    for i in range(len(idx)):
        arr[idx[i, 0], idx[i, 1], :] = means[i]

    block_mids = np.array([(b["start"] + b["end"]) / 2 / 1e6 for b in blocks])
    edge_ok = (block_mids > EDGE_TRIM_MB) & (block_mids < block_mids.max() - EDGE_TRIM_MB)

    # Accessibility
    acc_file = RESULTS_DIR / "accessibility_100kb.npz"
    if acc_file.exists():
        acc = np.load(acc_file)
        key = f"{arm}_frac"
        if key in acc:
            acc_frac = acc[key]
            acc_ok = np.array([
                acc_frac[i] >= DEMOG_MIN_ACC if i < len(acc_frac) else False
                for i in range(n_blocks)
            ])
        else:
            acc_ok = np.ones(n_blocks, dtype=bool)
    else:
        acc_ok = np.ones(n_blocks, dtype=bool)

    keep = edge_ok & acc_ok
    masked = arr[keep]
    return np.exp(masked[~np.isnan(masked)])


def coalescence_rates(tmrcas, time_windows):
    counts, _ = np.histogram(tmrcas, bins=time_windows)
    total = len(tmrcas)
    if total == 0:
        return np.zeros(len(time_windows) - 1)
    cum = np.cumsum(counts)
    surviving = total - np.concatenate([[0], cum[:-1]])
    widths = np.diff(time_windows)
    rates = np.where(
        (surviving > 0) & (widths > 0),
        (counts / surviving) / widths,
        0.0,
    )
    return rates


# ---------------------------------------------------------------------------
# Compute IICR for Burkina Faso, 3L and 3R
# ---------------------------------------------------------------------------

n_time_bins = 40
time_windows = np.logspace(2, 7, n_time_bins + 1)
time_windows[0] = 0.0
time_mids = np.sqrt(time_windows[:-1] * time_windows[1:])
time_mids[0] = time_windows[1] / 2

arm_colors = {
    "3L": "#4CAF50",
    "3R": "#E53935",
    "2L": "#2196F3",
    "2R": "#FF9800",
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for arm in ["3L", "3R"]:
    data = load_group("Burkina_Faso", arm, "all", "intra")
    if data is None:
        print(f"  No data for {arm}")
        continue

    tmrcas_gen = _get_masked_tmrcas(data, arm)
    rates = coalescence_rates(tmrcas_gen, time_windows)
    with np.errstate(divide="ignore", invalid="ignore"):
        iicr = np.where(rates > 0, 1.0 / (2.0 * rates), np.nan)

    # Convert to absolute units
    time_ka = time_mids / GEN_PER_YEAR / 1000  # generations -> ka
    ne_millions = iicr / 1e6

    # Mask out NaN / zero for clean plotting
    valid = np.isfinite(ne_millions) & (ne_millions > 0)
    ax.plot(time_ka[valid], ne_millions[valid],
            lw=2.5, color=arm_colors[arm], alpha=0.85,
            label=f"BF gambiae — {arm}")

# Style to match Stairway Plot
ax.set_xscale("log")
ax.set_yscale("log")

# Y axis: Ne in millions
ax.set_ylabel(r"$N_e$ (millions)", fontsize=13)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))
ax.set_ylim(0.05, 20)

# X axis: time in ka (present on left, past on right — same as stairway plot)
ax.set_xlabel("Time (ka)", fontsize=13)
ax.set_xlim(0.5, 100)

# Secondary y-axis: population size scaled as 4*mu*Ne (like stairway plot left axis)
ax2 = ax.twinx()
# 4*mu*Ne where Ne is in the same range as the primary axis
# At mu_mid: 4 * 3.5e-9 * Ne  =>  for Ne=1e6: 0.014;  Ne=10e6: 0.14
ne_low, ne_high = ax.get_ylim()  # in millions
scaled_low = 4 * MU_MID * ne_low * 1e6
scaled_high = 4 * MU_MID * ne_high * 1e6
ax2.set_yscale("log")
ax2.set_ylim(scaled_low, scaled_high)
ax2.set_ylabel(r"Population size (scaled in units of $4\mu N_e$)", fontsize=11)

ax.legend(fontsize=11, loc="upper right")
ax.set_title("Burkina Faso — fastcxt IICR (Stairway-Plot scale)\n"
             rf"$\mu$ = {MU_MID:.1e}/bp/gen, {GEN_PER_YEAR} gen/yr",
             fontsize=13)
ax.grid(True, which="both", alpha=0.15)

plt.tight_layout()
out = Path(__file__).parent / "demography" / "stairway_style_burkina_faso.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"Saved to {out}")
plt.close()

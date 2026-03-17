#!/usr/bin/env python3
"""Build composite paper figures merging human + AnoGam panels.

Outputs go to experiment/paper_figures/ and are referenced directly from paper.tex.

  fig2_combined.png  -- 2x2 scatter (3 human + 1 AnoGam)
  fig3_combined.png  -- 2 rows genome (1 human pair + 1 AnoGam pair), matched y-axis
  figS1_combined.png -- training curves stacked
  figS2_combined.png -- residuals stacked

Run from repo root:
  .venv/bin/python experiment4/make_paper_composites.py
"""
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
HUMAN = REPO / "experiment" / "paper_figures"
MOSQ = REPO / "experiment_mossies" / "paper_figures"
OUT = HUMAN


def _resize_width(im: Image.Image, target_w: int) -> Image.Image:
    ratio = target_w / im.width
    return im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)


def _resize_height(im: Image.Image, target_h: int) -> Image.Image:
    ratio = target_h / im.height
    return im.resize((int(im.width * ratio), target_h), Image.LANCZOS)


def fig2_scatter():
    """2x2 grid: crop the 1x3 human scatter into 3 panels, add AnoGam as 4th."""
    human = Image.open(HUMAN / "fig2_scatter.png")
    anogam = Image.open(MOSQ / "scatter_anogam.png")

    hw, hh = human.size
    panel_w = hw // 3

    panels = []
    for i in range(3):
        box = (i * panel_w, 0, (i + 1) * panel_w, hh)
        panels.append(human.crop(box))

    # Resize AnoGam to match the panel dimensions
    panels.append(_resize_height(anogam, hh))

    # All panels same height (hh). Find max width for uniform grid.
    max_w = max(p.width for p in panels)
    gap = 20

    canvas_w = max_w * 2 + gap
    canvas_h = hh * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    positions = [(0, 0), (max_w + gap, 0), (0, hh + gap), (max_w + gap, hh + gap)]
    for p, (x, y) in zip(panels, positions):
        # Center panels that are narrower
        x_off = (max_w - p.width) // 2
        canvas.paste(p, (x + x_off, y))

    canvas.save(OUT / "fig2_combined.png", "PNG")
    print(f"  fig2_combined.png ({canvas.size[0]}x{canvas.size[1]})")


def fig3_genome():
    """Two rows: first human pair (top row of fig3) + first AnoGam pair, same width."""
    human = Image.open(HUMAN / "fig3_tmrca_genome.png")
    anogam = Image.open(MOSQ / "genome_anogam.png")

    # Human: 3 rows -> crop top row
    hrow = human.height // 3
    human_top = human.crop((0, 0, human.width, hrow))

    # AnoGam: 3 rows -> crop top row
    arow = anogam.height // 3
    anogam_top = anogam.crop((0, 0, anogam.width, arow))

    target_w = human.width
    human_top = _resize_width(human_top, target_w)
    anogam_top = _resize_width(anogam_top, target_w)

    gap = 10
    canvas_h = human_top.height + anogam_top.height + gap
    canvas = Image.new("RGB", (target_w, canvas_h), "white")
    canvas.paste(human_top, (0, 0))
    canvas.paste(anogam_top, (0, human_top.height + gap))

    canvas.save(OUT / "fig3_combined.png", "PNG")
    print(f"  fig3_combined.png ({canvas.size[0]}x{canvas.size[1]})")


def figs1_training():
    """Stack human + AnoGam training curves."""
    human = Image.open(HUMAN / "figS1_training_curves.png")
    anogam = Image.open(MOSQ / "training_anogam.png")

    target_w = human.width
    anogam = _resize_width(anogam, target_w)

    gap = 10
    canvas_h = human.height + anogam.height + gap
    canvas = Image.new("RGB", (target_w, canvas_h), "white")
    canvas.paste(human, (0, 0))
    canvas.paste(anogam, (0, human.height + gap))

    canvas.save(OUT / "figS1_combined.png", "PNG")
    print(f"  figS1_combined.png ({canvas.size[0]}x{canvas.size[1]})")


def figs2_residuals():
    """Stack human residuals + AnoGam residuals."""
    human = Image.open(HUMAN / "figS2_residuals.png")
    anogam = Image.open(MOSQ / "residuals_anogam.png")

    target_w = human.width
    anogam = _resize_width(anogam, target_w)

    gap = 10
    canvas_h = human.height + anogam.height + gap
    canvas = Image.new("RGB", (target_w, canvas_h), "white")
    canvas.paste(human, (0, 0))
    canvas.paste(anogam, (0, human.height + gap))

    canvas.save(OUT / "figS2_combined.png", "PNG")
    print(f"  figS2_combined.png ({canvas.size[0]}x{canvas.size[1]})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig2_scatter()
    fig3_genome()
    figs1_training()
    figs2_residuals()
    print(f"\nAll composites in {OUT}/")


if __name__ == "__main__":
    main()

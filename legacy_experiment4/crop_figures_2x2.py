#!/usr/bin/env python3
"""Crop fig2_scatter (1x3) and fig3_tmrca_genome (3x1) into panels for 2x2 LaTeX layout.

Outputs to experiment/paper_figures/:
  fig2_scatter_a.png, fig2_scatter_b.png, fig2_scatter_c.png  (from fig2_scatter.png)
  fig3_genome_a.png, fig3_genome_b.png, fig3_genome_c.png     (from fig3_tmrca_genome.png)

Panel (d) for both figures is scatter_anogam.png and genome_anogam.png from
experiment_mossies/paper_figures/ (included directly in LaTeX).

Run from repo root:
  python experiment4/crop_figures_2x2.py
"""
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("pip install Pillow")

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "experiment" / "paper_figures"
OUT = SRC  # write panels next to originals
MOSQUITO = REPO / "experiment_mossies" / "paper_figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Figure 2: 1 row x 3 columns -> 3 panels (a,b,c). d = scatter_anogam (used as-is).
    fig2 = SRC / "fig2_scatter.png"
    if fig2.exists():
        im = Image.open(fig2)
        w, h = im.size
        third = w // 3
        for i, name in enumerate(["fig2_scatter_a", "fig2_scatter_b", "fig2_scatter_c"]):
            box = (i * third, 0, (i + 1) * third, h)
            im.crop(box).save(OUT / f"{name}.png", "PNG")
        print(f"  Cropped {fig2.name} -> fig2_scatter_a/b/c.png")
    else:
        print(f"  Skip (missing): {fig2}")

    # Figure 3: 3 rows x 1 column -> 3 panels (a,b,c). d = genome_anogam (used as-is).
    fig3 = SRC / "fig3_tmrca_genome.png"
    if fig3.exists():
        im = Image.open(fig3)
        w, h = im.size
        third = h // 3
        for i, name in enumerate(["fig3_genome_a", "fig3_genome_b", "fig3_genome_c"]):
            box = (0, i * third, w, (i + 1) * third)
            im.crop(box).save(OUT / f"{name}.png", "PNG")
        print(f"  Cropped {fig3.name} -> fig3_genome_a/b/c.png")
    else:
        print(f"  Skip (missing): {fig3}")

    # Figure S2 (residuals): 3 rows -> 3 panels (a,b,c). d = residuals_anogam (used as-is).
    figs2 = SRC / "figS2_residuals.png"
    if figs2.exists():
        im = Image.open(figs2)
        w, h = im.size
        third = h // 3
        for i, name in enumerate(["figS2_residuals_a", "figS2_residuals_b", "figS2_residuals_c"]):
            box = (0, i * third, w, (i + 1) * third)
            im.crop(box).save(OUT / f"{name}.png", "PNG")
        print(f"  Cropped {figs2.name} -> figS2_residuals_a/b/c.png")
    else:
        print(f"  Skip (missing): {figs2}")

    print("  Panel (d) for Fig 2: experiment_mossies/paper_figures/scatter_anogam.png")
    print("  Panel (d) for Fig 3: experiment_mossies/paper_figures/genome_anogam.png")
    print("  Panel (d) for Fig S2: experiment_mossies/paper_figures/residuals_anogam.png")


if __name__ == "__main__":
    main()

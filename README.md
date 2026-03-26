# fastcxt

**Fast pairwise coalescence time inference with Mamba state-space models.**

[![Documentation](https://readthedocs.org/projects/fastcxt/badge/?version=latest)](https://fastcxt.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Under active development.** APIs, documentation, and results may change without notice. Not yet recommended for production use.

fastcxt predicts pairwise time to most recent common ancestor (TMRCA) from genotype data using a **bidirectional Mamba encoder-decoder**. It replaces the autoregressive transformer from [cxt](https://github.com/kevinkorfmann/cxt) with a single-pass architecture that produces means and calibrated variances for all genomic windows in one forward pass.

**[Documentation](https://fastcxt.readthedocs.io)** | **[Figure Gallery](https://fastcxt.readthedocs.io/en/latest/gallery.html)** | **[API Reference](https://fastcxt.readthedocs.io/en/latest/api/index.html)**

---

## cxt vs fastcxt

| | **cxt** | **fastcxt** |
|---|---|---|
| Architecture | Decoder-only transformer (GPT-style) | Bidirectional Mamba encoder-decoder |
| Inference | Autoregressive: 15 samples x 500 decode steps | **Single forward pass** |
| Output | 324 discrete bins (classification) | Continuous (mu, log sigma^2) per window |
| Uncertainty | Monte Carlo (15 stochastic samples) | **Direct** (Gaussian NLL loss) |
| Mutation rate | Post-hoc bias correction | **FiLM conditioning** (learned per layer) |
| Sample sizes | Fixed at 50 (needs adapter) | **Any size up to 200** (zero-padded) |
| Forward passes per pair | ~15,000 | **1** |

## Quick start

```bash
pip install fastcxt
```

```python
from fastcxt.translate import translate_from_genotype_matrix

means, variances, index_map = translate_from_genotype_matrix(
    gm, positions, model,
    blocks=[(0, 100_000)],
    pivot_pairs=[(0, 1)],
    mutation_rate=3.5e-9,
    device="cuda:0",
)
# means: predicted log-TMRCA per window
# variances: calibrated uncertainty
# 95% CI: exp(means +/- 1.96 * sqrt(variances))
```

## How it works

1. **Build SFS features** -- Site frequency spectrum in XOR/XNOR channels from a genotype matrix
2. **Single forward pass** -- Bidirectional Mamba encoder reads the full sequence, decoder outputs (mu, log sigma^2) for every window
3. **FiLM conditioning** -- Mutation rate injected via learned scale/shift at each encoder layer
4. **Calibrated uncertainty** -- Beta-NLL loss directly models variance alongside the mean

## Ag1000G application

Comprehensive analysis of 16 *Anopheles gambiae* populations across Africa:

- **Karyotype-stratified inference** -- separate analysis for 2La/2Rb homozygous standard, heterozygous, and homozygous inverted individuals
- **Inversion detection** -- intra-individual pairs reveal deep coalescence inside In(2L)a and In(2R)b
- **Outlier scans** -- credible selection candidates filtered by accessibility mask, annotated with VectorBase genes
- **Demographic inference** -- IICR/Ne(t) estimation compared to stdpopsim reference
- **Geographic visualization** -- coalescence patterns projected onto Africa

See the **[Figure Gallery](https://fastcxt.readthedocs.io/en/latest/gallery.html)** for all results.

## Key modules

| Module | Description |
|---|---|
| `fastcxt.translate` | Inference API: `translate_from_genotype_matrix()`, `translate_from_ts()` |
| `fastcxt.model` | `FastCxtModel` -- bidirectional Mamba encoder-decoder |
| `fastcxt.config` | `FastCxtConfig`, `PRESETS`, `TrainingConfig` |
| `fastcxt.sfs` | SFS feature computation: `build_sfs_tensor()` |
| `fastcxt.atlas` | `TimeAtlas` -- genome-wide TMRCA storage and queries |
| `fastcxt.train` | `LitFastCxt` Lightning module + `fastcxt-train` CLI |
| `fastcxt.simulate` | Scenario registry + `fastcxt-simulate` CLI |
| `fastcxt.mosquito` | *A. gambiae* analysis protocol, accessibility masks |

## Citation

If you use fastcxt in your research, please cite:

```
@article{korfmann2026fastcxt,
  title={fastcxt: Fast pairwise coalescence time inference with Mamba state-space models},
  author={Korfmann, Kevin},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

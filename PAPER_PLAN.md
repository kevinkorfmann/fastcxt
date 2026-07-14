# fastcxt — 4-species large-Ne genealogy-atlas paper: progress checklist

Living tracker for the paper. `[x]` done · `[~]` in progress · `[ ]` todo.
Full plan: `~/.claude/plans/delightful-swinging-bonbon.md`. Work on branch `pivot/unpolarized-multiallelic`; compute on betty via SLURM.

**Scope:** 4 species = mosquito (`AnoGam`), *D. melanogaster* (`DroMel`), *Heliconius melpomene* (`HelMel`), honeybee (`ApiMel`); barnacle → appendix. Node-time model = primary method. SINGER\* head-to-head. Genome-wide atlases.

---

## Foundation — pivot validated (DONE, previous session)
- [x] **A. Folded/unpolarized SFS** (dual-mode) — benchmark: ~6% RMSE cost, 95% calibration preserved
- [x] **B1. Multi-allelic decompose** — validated on real DGN multi-allelic data (7,339 sites; +14,893 cols vs drop)
- [x] Correctness fixes — inference padding to `max_samples`; `PairDataset` double-`log1p` removed; `--population-size` CLI
- [x] Node-time model **sim-validated** (r=0.90 > tsdate 0.79) via `node_time_smoke`
- [x] Real-data ingested — mosquito Ag3.0 (316×45,981) + Drosophila DGN (197×118,262)
- [x] Real **tsdate genealogy atlases** (both species; π-validated; ~18× genome-wide variation)
- [x] **Diversity-amortized model validated end-to-end** (`atlas_v3`): real TMRCA within ~5–15% of π/tsdate across ~70× scale (mosq telo 30.5k, central 489k, Dros 1.84M)
- [x] Figures: `atlas_figures_v2.png`, `calibration_figure.png`
- [x] Key lesson recorded: π-check real-data scale; single-Ne amortized model regurgitates its prior OOD → fix = diversity-grid training

---

## P1 — Node-time model on real data  *(primary method)*
- [ ] Locate/confirm reusable node-time pipeline (`experiment_mosquito_poc/{preprocess_node_times,train_node_time,benchmark_scaling}.py`) + any trained node-time checkpoint
- [ ] Productionize into package: `LitNodeTime`/`LitHybridNodeTime` (train.py), node-time preprocessing (preprocess.py), `NodeTimeDataset` (dataset.py), `--model-type {pairwise,node-time,hybrid}`
- [ ] Train a HybridNodeTimeModel on the curriculum (folded full-sample SFS + topology)
- [ ] Real-data node-time inference: tsinfer topology → `extract_topology_features` → model → `lca_lookup_batch`, on mosquito (both regions) + Drosophila
- [ ] Compare vs tsdate + vs pairwise fastcxt + vs π; **show improved window-level resolution** (std(log-TMRCA) up from ~0.01–0.06)

## P2 — Folded-vs-polarized on real Drosophila  *(honest headline benchmark)*
- [ ] Ingest D. simulans outgroup (`DGN11/simulans_sequences.tar.gz`), align to Chr2L coords
- [ ] Polarize DGN Drosophila SNPs (ancestral = simulans base) → polarized genotype matrix
- [ ] Build polarized (unfolded) vs folded feature sets on identical sites
- [ ] Compare TMRCA accuracy + 95% CI coverage vs tsdate/π; quantify real-data folding cost

## P3 — Proper training run (curriculum + species)
- [~] Extend `simulate.py`: add `HelMel`/`ApiMel` to registry (done; set const Ne since no stdpopsim demog models); verify multi-allelic via custom scenarios; explicit μ prior
- [ ] Build curriculum: diversity grid (Ne 1e4..1e7 × μ) ∪ stdpopsim demographies (AnoGam/DroMel + HelMel/ApiMel maps) ∪ folded ∪ multi-allelic; polarized twin for P2
- [ ] Train production model (bigger/longer, torch.compile) — GPU long-pole
- [ ] Re-verify: π-calibration ~1.0× across all species/regions, CI coverage on real data, resolution vs tsdate

## P4 — Four-species genome-wide atlases
- [ ] Package VCF/zarr loader (currently ad hoc in scripts)
- [ ] Genome-wide `TimeAtlas` zarr/HDF5 backend + node-time storage; fix `pair_index` scan
- [ ] Mosquito genome-wide atlas (reuse `experiment_burkina_faso/build_atlas.py`)
- [ ] Drosophila genome-wide atlas (all arms)
- [ ] Heliconius atlas (ENA data — variant calling; **long pole**, start early)
- [ ] Honeybee atlas (Wallberg/ENA; haplodiploid free phasing)
- [ ] Per-species biology + downloadable atlas release

## P5 — SINGER\* head-to-head benchmark
- [ ] Build SINGER\* on betty (`~/Projects/SINGER` fork w/ DEER parallel-MCMC)
- [ ] Accuracy vs π/true on shared sims; accuracy on a real region
- [ ] Wall-clock/scaling vs fastcxt (pairwise + node-time) + tsdate; show MCMC crossover at large Ne

## P6 — Finalize tsinfer/tsdate tuning
- [ ] Fix KC-distance NaN (multi-root trees) in `experiment_tsinfer_tuning/`
- [ ] Run real-mode mismatch_ratio sweep per species; record best settings (tsdate μ=3.5e-9 for AnoGam)

## P7 — Reproducibility, docs, manuscript
- [ ] Package scratchpad scripts into repo (config-driven paths); commit GPU-env recipe
- [ ] Docs: folded/multi-allelic/node-time sections; fix `mosquito_protocol.rst` 1Mb-vs-100kb; update `training.rst` presets
- [ ] Create `paper/` IMRAD manuscript + figure-assembly pipeline
- [ ] Open PR to merge `pivot/unpolarized-multiallelic`

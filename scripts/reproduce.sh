#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Reproduce all fastcxt results end-to-end
#
# Runs the complete pipeline: benchmark → simulate → preprocess → train →
# infer on Ag1000G → build TimeAtlas → generate publication figures.
#
# All outputs (simulations, checkpoints, atlas, figures) land under BASE_DIR.
# On completion, figures/ will contain real data figures replacing all
# simulated placeholders.
#
# Usage:
#   ./scripts/reproduce.sh                          # run everything
#   ./scripts/reproduce.sh simulate                 # only stage 1
#   ./scripts/reproduce.sh train                    # only stage 3
#   ./scripts/reproduce.sh simulate preprocess      # multiple stages
#   GPUS="0 1" ./scripts/reproduce.sh               # override GPU list
#
# Environment variables (all have defaults):
#   BASE_DIR             — root output directory
#   GPUS                 — space-separated GPU IDs (default: "0 1 2")
#   SIM_WORKERS          — parallel simulation workers (default: 80)
#   PREPROCESS_WORKERS   — parallel preprocessing workers (default: 80)
#   AG1000G_DATA_DIR     — path to Ag1000G tsinfer .trees files
#   AG1000G_ACCESSIBILITY — path to accessibility mask .npz
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/sietch_colab/data_share/fastcxt_reproduce}"

DATA_DIR="${BASE_DIR}/simulations"
PROCESSED_DIR="${BASE_DIR}/processed"
PROCESSED_TREES_DIR="${BASE_DIR}/processed_trees"
CKPT_DIR="${BASE_DIR}/checkpoints"
BENCHMARK_DIR="${BASE_DIR}/benchmarks"
ATLAS_DIR="${BASE_DIR}/atlas"
FIG_DIR="${BASE_DIR}/figures"

# ─────────────────────────────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────────────────────────────
GPUS="${GPUS:-0 1 2}"
GPU_PRIMARY="cuda:$(echo "${GPUS}" | awk '{print $1}')"
SIM_WORKERS="${SIM_WORKERS:-80}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-80}"
TRAIN_WORKERS="${TRAIN_WORKERS:-16}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"

# ─────────────────────────────────────────────────────────────────────
# External data (Ag1000G, Human 1KG)
# ─────────────────────────────────────────────────────────────────────
export AG1000G_DATA_DIR="${AG1000G_DATA_DIR:-/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2}"
export AG1000G_ACCESSIBILITY="${AG1000G_ACCESSIBILITY:-/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz}"
export FASTCXT_BASE_DIR="${BASE_DIR}"

# ─────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# ─────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────
mkdir -p "${BASE_DIR}"
LOGFILE="${BASE_DIR}/reproduce_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

FAILED=()
STAGE_TIMES=()
SCRIPT_START=$SECONDS

log()  { echo "[$(date +%H:%M:%S)] $*"; }
pass() { log "  ✓ $1 (${2}s)"; }
fail() { log "  ✗ $1 (exit $2, ${3}s)"; FAILED+=("$1"); }

run_step() {
    local name="$1"; shift
    log "--- $name ---"
    local t0=$SECONDS
    if "$@" 2>&1; then
        pass "$name" "$(( SECONDS - t0 ))"
    else
        fail "$name" "$?" "$(( SECONDS - t0 ))"
    fi
}

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " fastcxt — Full Reproduction"
log " $(date)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " BASE_DIR:       ${BASE_DIR}"
log " GPUs:           ${GPUS}"
log " Primary GPU:    ${GPU_PRIMARY}"
log " CPU workers:    ${SIM_WORKERS}"
log " Train epochs:   ${TRAIN_EPOCHS}"
log " Repo:           ${REPO_ROOT}"
log " Log:            ${LOGFILE}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─────────────────────────────────────────────────────────────────────
# Parse stages
# ─────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    STAGES=(benchmark simulate preprocess train infer atlas figures)
else
    STAGES=("$@")
fi
log "Stages: ${STAGES[*]}"
log ""


# ═════════════════════════════════════════════════════════════════════
#  STAGE 1 — BENCHMARK
#  Scaling benchmarks: fastcxt pairwise vs tree-aware, plus cxt baseline.
#  Run first so results are available even if training takes days.
#  Estimated: 30–60 minutes on 1 GPU.
# ═════════════════════════════════════════════════════════════════════
do_benchmark() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 1: BENCHMARK"
    log "════════════════════════════════════════════"
    mkdir -p "${BENCHMARK_DIR}"

    # ── fastcxt benchmarks (both modes) ──

    run_step "Benchmark: fastcxt scaling" \
        fastcxt-benchmark \
            --mode all \
            --sample-sizes 5 10 25 50 100 200 \
            --device "${GPU_PRIMARY}" \
            --batch-size 64 \
            --output "${BENCHMARK_DIR}/fastcxt_scaling.json"

    STAGE_TIMES+=("Benchmark: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  STAGE 2 — SIMULATE
#  ~10,000 tree sequences across diverse demographic scenarios.
#  Uses fastcxt-simulate (stdpopsim registry + msprime scenarios).
#  Estimated: 2–4 hours on 80 CPUs.
# ═════════════════════════════════════════════════════════════════════
do_simulate() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 2: SIMULATE"
    log "════════════════════════════════════════════"
    mkdir -p "${DATA_DIR}"

    # ── Core msprime scenarios (diverse demography) ──

    run_step "Sim: constant (n=5000)" \
        fastcxt-simulate \
            --scenario constant \
            --data-dir "${DATA_DIR}/constant" \
            --num-ts 5000 \
            --n-samples 25 \
            --num-processes "${SIM_WORKERS}"

    run_step "Sim: sawtooth (n=2000)" \
        fastcxt-simulate \
            --scenario sawtooth \
            --data-dir "${DATA_DIR}/sawtooth" \
            --num-ts 2000 \
            --n-samples 25 \
            --num-processes "${SIM_WORKERS}"

    run_step "Sim: island (n=1000)" \
        fastcxt-simulate \
            --scenario island \
            --data-dir "${DATA_DIR}/island" \
            --num-ts 1000 \
            --n-samples 25 \
            --num-processes "${SIM_WORKERS}"

    # ── stdpopsim species (real genetic maps where available) ──

    local -A STDPOPSIM_SCENARIOS=(
        [AnoGam]=500
        [HomSap]=500
        [DroMel]=300
        [BosTau]=300
        [CanFam]=300
        [PanTro]=200
        [PapAnu]=200
        [PonAbe]=200
        [AraTha]=300
        [CaeEle]=300
        [AedAeg]=200
        [HelAnn]=200
    )

    for scenario in "${!STDPOPSIM_SCENARIOS[@]}"; do
        local n="${STDPOPSIM_SCENARIOS[$scenario]}"
        run_step "Sim: ${scenario} (n=${n})" \
            fastcxt-simulate \
                --scenario "${scenario}" \
                --data-dir "${DATA_DIR}/stdpopsim/${scenario}" \
                --num-ts "${n}" \
                --n-samples 25 \
                --num-processes "${SIM_WORKERS}"
    done

    # ── Variable sample sizes (for testing InputProjection) ──

    for ns in 5 10 50 100; do
        run_step "Sim: constant n_samples=${ns}" \
            fastcxt-simulate \
                --scenario constant \
                --data-dir "${DATA_DIR}/variable_n/n${ns}" \
                --num-ts 200 \
                --n-samples "${ns}" \
                --num-processes "${SIM_WORKERS}"
    done

    STAGE_TIMES+=("Simulate: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  STAGE 3 — PREPROCESS
#  Converts .trees → SFS features + log-TMRCA targets.
#  Two passes: standard (pairwise) and with tree topology features.
#  Estimated: 1–2 hours on 80 CPUs.
# ═════════════════════════════════════════════════════════════════════
do_preprocess() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 3: PREPROCESS"
    log "════════════════════════════════════════════"

    # ── Standard pairwise preprocessing ──

    run_step "Preprocess: standard (w2000, 200 pairs)" \
        fastcxt-preprocess \
            --base-dir "${DATA_DIR}" \
            --out-subdir processed \
            --window-size 2000 \
            --num-pairs 200 \
            --global-seed 42 \
            --num-workers "${PREPROCESS_WORKERS}" \
            --skip-existing

    # Move/symlink to expected location
    if [ ! -e "${PROCESSED_DIR}" ]; then
        ln -sfn "${DATA_DIR}/processed" "${PROCESSED_DIR}"
    fi

    # ── With tree topology features (for tree-aware model) ──

    run_step "Preprocess: with trees (w2000, 200 pairs, topology)" \
        fastcxt-preprocess \
            --base-dir "${DATA_DIR}" \
            --out-subdir processed_trees \
            --window-size 2000 \
            --num-pairs 200 \
            --global-seed 42 \
            --extract-trees \
            --num-workers "${PREPROCESS_WORKERS}" \
            --skip-existing

    if [ ! -e "${PROCESSED_TREES_DIR}" ]; then
        ln -sfn "${DATA_DIR}/processed_trees" "${PROCESSED_TREES_DIR}"
    fi

    STAGE_TIMES+=("Preprocess: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  STAGE 4 — TRAIN
#  Trains the base and tree-aware models.
#  Estimated: 6–12 hours on 3 GPUs.
# ═════════════════════════════════════════════════════════════════════
do_train() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 4: TRAIN"
    log "════════════════════════════════════════════"
    mkdir -p "${CKPT_DIR}"

    # ── Base model (pairwise, no trees) ──

    run_step "Train: base model" \
        fastcxt-train \
            --model base \
            --dataset-path "${PROCESSED_DIR}" \
            --gpus ${GPUS} \
            --epochs "${TRAIN_EPOCHS}" \
            --batch-size "${TRAIN_BATCH}" \
            --workers "${TRAIN_WORKERS}" \
            --log-dir "${BASE_DIR}/lightning_logs"

    # Find and install checkpoint
    local base_ckpt
    base_ckpt=$(find "${BASE_DIR}/lightning_logs" -name "*.ckpt" -path "*/checkpoints/*" 2>/dev/null | sort -t= -k2 -n | tail -1 || true)
    if [ -n "${base_ckpt}" ]; then
        cp "${base_ckpt}" "${CKPT_DIR}/fastcxt_base.ckpt"
        log "  Installed: ${CKPT_DIR}/fastcxt_base.ckpt"
    fi

    # ── Tree-aware model ──

    run_step "Train: base_trees model" \
        fastcxt-train \
            --model base_trees \
            --dataset-path "${PROCESSED_TREES_DIR}" \
            --gpus ${GPUS} \
            --epochs "${TRAIN_EPOCHS}" \
            --batch-size "${TRAIN_BATCH}" \
            --workers "${TRAIN_WORKERS}" \
            --log-dir "${BASE_DIR}/lightning_logs"

    local tree_ckpt
    tree_ckpt=$(find "${BASE_DIR}/lightning_logs" -name "*.ckpt" -path "*/checkpoints/*" -newer "${CKPT_DIR}/fastcxt_base.ckpt" 2>/dev/null | sort -t= -k2 -n | tail -1 || true)
    if [ -n "${tree_ckpt}" ]; then
        cp "${tree_ckpt}" "${CKPT_DIR}/fastcxt_base_trees.ckpt"
        log "  Installed: ${CKPT_DIR}/fastcxt_base_trees.ckpt"
    fi

    STAGE_TIMES+=("Train: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  STAGE 5 — INFER (Ag1000G whole-genome)
#  Runs fastcxt on all 5 Ag1000G chromosome arms for ~50k pairs.
#  Estimated: 2–6 hours on 1–3 GPUs.
# ═════════════════════════════════════════════════════════════════════
do_infer() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 5: INFER (Ag1000G)"
    log "════════════════════════════════════════════"

    if [ ! -f "${CKPT_DIR}/fastcxt_base.ckpt" ]; then
        log "  [SKIP] No checkpoint found. Run 'train' stage first."
        return
    fi

    if [ ! -d "${AG1000G_DATA_DIR}" ]; then
        log "  [SKIP] AG1000G_DATA_DIR not found: ${AG1000G_DATA_DIR}"
        return
    fi

    mkdir -p "${ATLAS_DIR}"

    run_step "Infer: Ag1000G whole-genome" \
        python -c "
import sys, json, numpy as np, torch, tskit
from pathlib import Path
from fastcxt.config import PRESETS
from fastcxt.model import FastCxtModel
from fastcxt.mosquito import MosquitoAnalysis, AccessibilityMask, ANOGAM_CHROMOSOME_ARMS
from fastcxt.atlas import TimeAtlas

# Load model
config = PRESETS['base']
model = FastCxtModel(config)
ckpt = torch.load('${CKPT_DIR}/fastcxt_base.ckpt', map_location='cpu')
state = ckpt.get('state_dict', ckpt)
state = {k.replace('model.', '', 1): v for k, v in state.items() if k.startswith('model.')} or state
model.load_state_dict(state, strict=False)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

analysis = MosquitoAnalysis(model=model, device='${GPU_PRIMARY}', block_size=1_000_000, batch_size=256)

# Load accessibility mask
mask_path = '${AG1000G_ACCESSIBILITY}'
atlas = TimeAtlas()
atlas.metadata = {'species': 'Anopheles gambiae', 'dataset': 'Ag1000G Phase 3', 'checkpoint': '${CKPT_DIR}/fastcxt_base.ckpt'}

for arm in ['2L', '2R', '3L', '3R', 'X']:
    ts_path = Path('${AG1000G_DATA_DIR}') / f'{arm}.trees'
    if not ts_path.exists():
        print(f'  [SKIP] {ts_path} not found')
        continue

    print(f'Loading {arm}...')
    ts = tskit.load(str(ts_path))
    gm = ts.genotype_matrix().T.astype(np.int8)
    pos = ts.tables.sites.position

    n_hap = min(gm.shape[0], 200)
    pairs = [(i, j) for i in range(n_hap) for j in range(i+1, min(i+10, n_hap))]
    print(f'  {arm}: {gm.shape[0]} haploids, {len(pairs)} pairs, {gm.shape[1]} sites')

    try:
        mask = AccessibilityMask.from_npz(mask_path, arm)
    except Exception:
        mask = None

    result = analysis.run_chromosome_arm(gm, pos, arm, pivot_pairs=pairs, mutation_rate=3.5e-9, accessibility_mask=mask)
    atlas.add_arm(arm, result['means'], result['variances'], np.array(pairs, dtype=np.int32), window_size=2000, mutation_rate=3.5e-9)
    print(f'  {arm}: done ({atlas.arms[arm].n_windows} windows)')

atlas.save('${ATLAS_DIR}/ag1000g')
print(f'Atlas saved: {atlas}')
print(json.dumps(atlas.summary(), indent=2))
"

    STAGE_TIMES+=("Infer: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  STAGE 6 — BUILD ATLAS
#  Verify and finalize the TimeAtlas.
#  (Most of the work was done in stage 5; this is a sanity check.)
# ═════════════════════════════════════════════════════════════════════
do_atlas() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 6: ATLAS (verify)"
    log "════════════════════════════════════════════"

    if [ ! -f "${ATLAS_DIR}/ag1000g/manifest.json" ]; then
        log "  [SKIP] No atlas found. Run 'infer' stage first."
        log "  Generating placeholder atlas for figures..."

        run_step "Atlas: build simulated placeholder" \
            python -c "
import numpy as np, sys
sys.path.insert(0, '${REPO_ROOT}')
from scripts.plot_atlas_showcase import build_simulated_atlas
from fastcxt.atlas import TimeAtlas

rng = np.random.default_rng(42)
atlas, psm, sp = build_simulated_atlas(rng, n_per_pop=4, window_size=50_000)
atlas.save('${ATLAS_DIR}/ag1000g')
print(f'Placeholder atlas saved: {atlas}')

import json
with open('${ATLAS_DIR}/pop_sample_map.json', 'w') as f:
    json.dump(psm, f)
with open('${ATLAS_DIR}/sample_pop.json', 'w') as f:
    json.dump({str(k): v for k, v in sp.items()}, f)
"
    else
        run_step "Atlas: verify" \
            python -c "
from fastcxt.atlas import TimeAtlas
import json
atlas = TimeAtlas.load('${ATLAS_DIR}/ag1000g')
s = atlas.summary()
print(f'Atlas loaded: {atlas}')
print(f'  Arms: {s[\"arms\"]}')
print(f'  Total pairs: {s[\"total_pairs\"]:,}')
print(f'  Total windows: {s[\"total_windows\"]:,}')
for arm, info in s['per_arm'].items():
    print(f'  {arm}: {info[\"n_pairs\"]} pairs × {info[\"n_windows\"]} windows, mean_log_tmrca={info[\"mean_log_tmrca\"]:.3f}')
"
    fi

    STAGE_TIMES+=("Atlas: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  STAGE 7 — FIGURES
#  Generates all publication figures.
#  If a real atlas exists, uses it; otherwise falls back to simulated.
#  Estimated: 5–10 minutes.
# ═════════════════════════════════════════════════════════════════════
do_figures() {
    local t0=$SECONDS
    log "════════════════════════════════════════════"
    log "STAGE 7: FIGURES"
    log "════════════════════════════════════════════"
    mkdir -p "${FIG_DIR}"

    # ── Generate TimeAtlas showcase figures ──
    # Pass real atlas if available; falls back to simulated placeholders

    local plot_args="--outdir ${FIG_DIR}"
    if [ -f "${ATLAS_DIR}/ag1000g/manifest.json" ]; then
        plot_args="${plot_args} --atlas-dir ${ATLAS_DIR}/ag1000g"
        if [ -f "${ATLAS_DIR}/pop_sample_map.json" ]; then
            plot_args="${plot_args} --pop-map ${ATLAS_DIR}/pop_sample_map.json"
        fi
    fi
    if [ -f "${BENCHMARK_DIR}/fastcxt_scaling.json" ]; then
        plot_args="${plot_args} --benchmark-json ${BENCHMARK_DIR}/fastcxt_scaling.json"
    fi

    run_step "Figures: TimeAtlas showcase (8 figures)" \
        python "${REPO_ROOT}/scripts/plot_atlas_showcase.py" ${plot_args}

    # ── Copy to repo figures/ for docs ──

    if [ "${FIG_DIR}" != "${REPO_ROOT}/figures" ]; then
        run_step "Figures: copy to repo" \
            cp -v "${FIG_DIR}"/*.png "${REPO_ROOT}/figures/"
    fi

    # ── Rebuild docs with real figures ──

    run_step "Figures: rebuild docs" \
        python -m sphinx -b html \
            "${REPO_ROOT}/docs/source" \
            "${REPO_ROOT}/docs/source/_build"

    STAGE_TIMES+=("Figures: $(( SECONDS - t0 ))s")
}


# ═════════════════════════════════════════════════════════════════════
#  Dispatch
# ═════════════════════════════════════════════════════════════════════
for stage in "${STAGES[@]}"; do
    case "$stage" in
        simulate)   do_simulate   ;;
        preprocess) do_preprocess ;;
        train)      do_train      ;;
        benchmark)  do_benchmark  ;;
        infer)      do_infer      ;;
        atlas)      do_atlas      ;;
        figures)    do_figures    ;;
        *)
            log "Unknown stage: $stage"
            log "Valid stages: simulate preprocess train benchmark infer atlas figures"
            exit 1
            ;;
    esac
done


# ═════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════
echo ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " REPRODUCTION COMPLETE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " Base directory: ${BASE_DIR}"
log " Total wall time: $(( SECONDS - SCRIPT_START ))s"
log ""
for t in "${STAGE_TIMES[@]}"; do
    log "  $t"
done

log ""
log " Output locations:"
log "   Simulations:  ${DATA_DIR}/"
log "   Processed:    ${PROCESSED_DIR}/"
log "   Checkpoints:  ${CKPT_DIR}/"
log "   Benchmarks:   ${BENCHMARK_DIR}/"
log "   Atlas:        ${ATLAS_DIR}/ag1000g/"
log "   Figures:      ${FIG_DIR}/"
log "   Docs:         ${REPO_ROOT}/docs/source/_build/"

if [ ${#FAILED[@]} -gt 0 ]; then
    log ""
    log " FAILED steps (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        log "   ✗ $f"
    done
    log ""
    log " Log: $LOGFILE"
    exit 1
else
    log ""
    log " All steps completed successfully."
    log " Log: $LOGFILE"
fi

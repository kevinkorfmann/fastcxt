#!/usr/bin/env bash
set -euo pipefail
# Simulate AnoGam (Anopheles gambiae) tree sequences via stdpopsim
# 100 kb sequence, 200 bp windows → 500 windows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMS_DIR="$SCRIPT_DIR/sims"
NUM_TS=200
NUM_PROCS=${NUM_PROCS:-8}

if [ -d "$SIMS_DIR" ] && [ "${SKIP_EXISTING:-}" = "1" ]; then
    echo "Sims directory exists, skipping (SKIP_EXISTING=1)"
    exit 0
fi

for N in 10 25 50 100 200; do
    echo "Simulating AnoGam n_samples=$N ($NUM_TS tree sequences) ..."
    fastcxt-simulate \
        --scenario AnoGam \
        --data-dir "$SIMS_DIR/n${N}" \
        --num-ts "$NUM_TS" \
        --n-samples "$N" \
        --sequence-length 100000 \
        --num-processes "$NUM_PROCS"
done

echo "Simulations complete: $SIMS_DIR/"

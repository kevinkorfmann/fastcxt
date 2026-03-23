#!/usr/bin/env bash
set -euo pipefail
# Simulate human-like tree sequences (constant Ne=2e4, 1Mb, multiple sample sizes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMS_DIR="$SCRIPT_DIR/sims"
NUM_TS=200
NUM_PROCS=${NUM_PROCS:-8}

if [ -d "$SIMS_DIR" ] && [ "${SKIP_EXISTING:-}" = "1" ]; then
    echo "Sims directory exists, skipping (SKIP_EXISTING=1)"
    exit 0
fi

for N in 10 25 50 100 200; do
    echo "Simulating n_samples=$N ($NUM_TS tree sequences) ..."
    fastcxt-simulate \
        --scenario constant \
        --data-dir "$SIMS_DIR/n${N}" \
        --num-ts "$NUM_TS" \
        --n-samples "$N" \
        --sequence-length 1e6 \
        --num-processes "$NUM_PROCS"
done

echo "Simulations complete: $SIMS_DIR/"

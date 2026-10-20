#!/usr/bin/env bash
# Reruns every experiment in the README, then rebuilds the tables.
#
# Pretrains six vision networks (about 3 minutes each on Apple MPS), then
# runs evaluate.py once per row, writing one JSON per row to results/.
# About 85 minutes in all on an M-series Mac, run one job at a time.
#
#   bash scripts/run_grid.sh            # everything
#   SKIP_PRETRAIN=1 bash scripts/run_grid.sh   # reuse checkpoints/
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DEV=${DEV:-mps}
CK=checkpoints

if [ -z "${SKIP_PRETRAIN:-}" ]; then
    # Seed 0 with bce is the shipped checkpoints/vc_pretrained.pth.tar.
    for s in 1 2; do
        $PY pretrain.py --device $DEV --seed $s --name vc_bce_s$s --filters-plot figures/vc_bce_s$s.png
    done
    for s in 0 1 2; do
        $PY pretrain.py --device $DEV --seed $s --loss mse --name vc_mse_s$s \
            --filters-plot figures/vc_mse_s$s.png
    done
    $PY pretrain.py --device $DEV --seed 0 --no-centre --name vc_nocentre \
        --filters-plot figures/vc_nocentre.png
fi

BCE3="$CK/vc_pretrained.pth.tar $CK/vc_bce_s1.pth.tar $CK/vc_bce_s2.pth.tar"
MSE3="$CK/vc_mse_s0.pth.tar $CK/vc_mse_s1.pth.tar $CK/vc_mse_s2.pth.tar"
ev() { $PY evaluate.py --device $DEV "$@"; }

# Matching direction and scale.
ev --out results/oneshot.json
ev --match test-to-study --out results/match_test_to_study.json
ev --normalise --out results/normalise.json
ev --normalise --match test-to-study --out results/normalise_test_to_study.json

# Normalising the VC output before the hippocampal parts.
ev --normalise-input --out results/normalise_input.json
ev --normalise-input --normalise --out results/normalise_input_normalise.json

# Pretraining: seeds and loss.
ev --checkpoint $BCE3 --out results/pretrain_seeds.json
ev --checkpoint $CK/vc_mse_s0.pth.tar --out results/mse.json
ev --checkpoint $CK/vc_mse_s0.pth.tar --normalise --out results/mse_normalise.json
ev --checkpoint $MSE3 --out results/mse_pretrain_seeds.json
ev --checkpoint $CK/vc_mse_s0.pth.tar --no-interest-filter --out results/mse_no_interest_filter.json
ev --checkpoint $CK/vc_mse_s0.pth.tar --normalise-input --normalise \
    --out results/mse_normalise_input_normalise.json

# Best configuration found, normalised matching (results/normalise.json),
# across pretraining seeds. It was picked by looking at the test scores
# above, so it is not a held-out result.
ev --normalise --checkpoint $BCE3 --out results/best_pretrain_seeds.json

# Ablations under the default settings.
ev --no-interest-filter --out results/ablation_no_interest_filter.json
ev --no-centre --checkpoint $CK/vc_nocentre.pth.tar --out results/ablation_no_centre.json
ev --ca3-rule hebbian --out results/ablation_hebbian.json

# One CPU seed, to check the device doesn't change the scores.
$PY evaluate.py --device cpu --seeds 1 --out results/cpu_seed0.json

$PY scripts/make_tables.py

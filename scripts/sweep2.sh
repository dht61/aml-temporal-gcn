#!/bin/bash
#SBATCH --job-name=tgcn_sweep2
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
# per_node LSTM (the only big win in sweep.sh) alone and combined with the small ones, 10 seeds each
set -eo pipefail
source scripts/common.sh
declare -A F=(
  [pn]="--lstm_mode per_node"
  [pn_wd]="--lstm_mode per_node --weight_decay 5e-4"
  [pn_und]="--lstm_mode per_node --edges undirected"
  [pn_wd_und]="--lstm_mode per_node --weight_decay 5e-4 --edges undirected"
  [pn_ref5]="--lstm_mode per_node --weight_decay 5e-4 --class_weights 0.7,0.3"
)
for v in pn pn_wd pn_und pn_wd_und pn_ref5; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    python run_supervised.py --out results/sweep2/${v}_seed$s --seed $s ${F[$v]}
  done
done
# determinism check: rerun seed 0 and compare fingerprints
python run_supervised.py --out results/det_pernode/pn_seed0_rep --seed 0 ${F[pn]}
python compare_runs.py results/sweep2/pn_seed0 results/det_pernode/pn_seed0_rep

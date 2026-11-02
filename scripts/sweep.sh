#!/bin/bash
#SBATCH --job-name=tgcn_sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
# one change at a time vs the literal model, seeds 0-2
# ref5 = the first author's earlier GCN recipe (wd 5e-4, class weights 0.7/0.3); cw37 = Weber et al.'s weights
set -eo pipefail
source scripts/common.sh
V=(wd5e-4 cw73 ref5 cw37 softmax_nll per_node undirected)
v=${V[$SLURM_ARRAY_TASK_ID]}
case $v in
  wd5e-4)      F="--weight_decay 5e-4" ;;
  cw73)        F="--class_weights 0.7,0.3" ;;
  ref5)        F="--weight_decay 5e-4 --class_weights 0.7,0.3" ;;
  cw37)        F="--class_weights 0.3,0.7" ;;
  softmax_nll) F="--output softmax_nll" ;;
  per_node)    F="--lstm_mode per_node" ;;
  undirected)  F="--edges undirected" ;;
esac
for s in 0 1 2; do
  python run_supervised.py --out results/sweep/${v}_seed$s --seed $s $F
done

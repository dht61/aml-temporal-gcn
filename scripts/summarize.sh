#!/bin/bash
#SBATCH --job-name=tgcn_summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source scripts/common.sh
python summarize_supervised.py --sweep sweep2 --ablation ablation_pernode --md results/summary.md
python summarize_supervised.py --md results/sweep/summary.md
python summarize_al.py --al al_pernode --sup sweep2/pn --sup_seq table1/model0

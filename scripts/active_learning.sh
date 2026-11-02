#!/bin/bash
#SBATCH --job-name=tgcn_al
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
# Fig. 4 / Tables 2-3: {MC-dropout (T=50), MC-AA (T=10)} x 4 acquisitions + random, seed 0
set -eo pipefail
source scripts/common.sh
ARMS=(mc_dropout:bald mc_dropout:entropy mc_dropout:variation_ratio mc_dropout:mean_std
      mc_aa:bald mc_aa:entropy mc_aa:variation_ratio mc_aa:mean_std random:random)
arm=${ARMS[$SLURM_ARRAY_TASK_ID]}
m=${arm%%:*}; q=${arm##*:}
python run_active_learning.py --out results/al_pernode/seed0/${m}_${q} --method $m --acquisition $q --seed 0 --lstm_mode per_node

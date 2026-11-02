#!/bin/bash
#SBATCH --job-name=tgcn_table1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
# Table 1 with the literal (seq_nodes) LSTM, 10 seeds. ~35 min per seed on 8 threads.
set -eo pipefail
source scripts/common.sh
python run_supervised.py --out results/table1/model0_seed${SLURM_ARRAY_TASK_ID} --seed ${SLURM_ARRAY_TASK_ID}

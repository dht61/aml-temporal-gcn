#!/bin/bash
#SBATCH --job-name=tgcn_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source scripts/common.sh
python prep_data.py

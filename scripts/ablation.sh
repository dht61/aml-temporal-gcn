#!/bin/bash
#SBATCH --job-name=tgcn_ablation
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out
# Tables 4-5 (per_node LSTM): model0-7 and K=1,2, 5 seeds each
set -eo pipefail
source scripts/common.sh
for v in model0 model1 model2 model3 model4 model5 model6 model7 K1 K2; do
  for s in 0 1 2 3 4; do
    case $v in
      K*) python run_supervised.py --out results/ablation_pernode/model0_${v}_seed$s --model model0 --K ${v#K} --seed $s --lstm_mode per_node ;;
      *)  python run_supervised.py --out results/ablation_pernode/${v}_seed$s --model $v --seed $s --lstm_mode per_node ;;
    esac
  done
done

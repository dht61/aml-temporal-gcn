#!/bin/bash
#SBATCH --job-name=tgcn_det
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# quick check that every code path is bitwise reproducible (short runs, each done twice)
set -eo pipefail
source scripts/common.sh
S=results/det_check
rc=0
for r in a b; do
  python run_supervised.py --out $S/sup_$r --epochs 3 --eval_every 1 --force
done
python compare_runs.py $S/sup_a $S/sup_b || rc=1
for m in mc_dropout:bald mc_aa:bald random:random; do
  for r in a b; do
    rm -rf $S/al_${m%%:*}_$r
    python run_active_learning.py --out $S/al_${m%%:*}_$r --method ${m%%:*} --acquisition ${m##*:} \
      --epochs 2 --max_iters 2 --T_dropout 3 --T_aa 3
  done
  python compare_runs.py $S/al_${m%%:*}_a $S/al_${m%%:*}_b || rc=1
done
echo "determinism check: $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc

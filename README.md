# aml-temporal-gcn

Reproduction of Alarab & Prakoonwit, *Graph-Based LSTM for Anti-money Laundering: Experimenting Temporal Graph
Convolutional Network with Bitcoin Data*, Neural Processing Letters 2023
([paper](https://doi.org/10.1007/s11063-022-10904-8)).

There's no official code, so this is written from the paper. The task is classifying Bitcoin transactions in the
Elliptic dataset as illicit or licit. It trains on time steps 1-34, tests on 35-49 and uses the local features only.
The model runs LSTM -> TAGConv -> TAGConv on each time step's graph. The second half of the paper is active learning
with MC-dropout and MC-AA uncertainty.

## Results

These are illicit-class scores on the test set, averaged over 10 seeds. The paper reports a single run.

| | Acc | Precision | Recall | F1 |
|---|---|---|---|---|
| paper | 97.7 | 92.7 | 71.3 | 80.6 |
| this repo | 97.50 ± 0.08 | 90.9 ± 2.2 | 68.4 ± 0.8 | 78.0 ± 0.4 |

The paper doesn't say how the LSTM is applied to the node feature matrix. Taken literally, PyTorch 1.11 treats the
matrix as one long sequence over all nodes. That drops F1 to 73.8, so the default here runs the LSTM per node
(`--lstm_mode per_node`). It's also a lot faster.

For active learning with MC-AA, labelling about 20% of the pool is enough to match the fully supervised accuracy.
Random sampling needs 87%. MC-dropout needs 20-47%, depending on the acquisition function.

![active learning curves](results/al_pernode/figs/fig4_accuracy.png)

Removing the LSTM hurts the most, as in the paper. The rest of the layer ablation didn't reproduce: simpler
variants, even a plain MLP, tie the full model, and K = 1, 2 and 3 hops are all within noise of each other.

Full tables are in `results/summary.md`, with per-run metrics alongside.

## Running

Everything runs on CPU. The scripts in `scripts/` are SLURM jobs, but `bash` works too.

```bash
bash scripts/setup_env.sh          # conda env, torch 1.11 + PyG 2.0.4
sbatch scripts/prep_data.sh        # downloads Elliptic
sbatch scripts/sweep2.sh           # main table, 10 seeds
sbatch scripts/ablation.sh
sbatch scripts/active_learning.sh
sbatch scripts/summarize.sh
```

To do a single run:

```bash
python run_supervised.py --out results/test --lstm_mode per_node --seed 0
```

Runs are deterministic on the same machine and thread count. Each one saves sha256 fingerprints of the model and
its predictions, which `compare_runs.py` checks. The data and model weights aren't in the repo.

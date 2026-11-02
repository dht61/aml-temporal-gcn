#!/bin/bash
# conda env with the paper's stack: torch 1.11.0 + PyG 2.0.4 (CPU builds).
# 1.11 is the first torch whose nn.LSTM takes a 2-D input, and PyG 2.0.4's TAGConv has one bias per hop.
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^aml_tgcn\s"; then
  conda create -y -n aml_tgcn -c conda-forge --override-channels python=3.10 pip
fi
conda activate aml_tgcn
PIN="numpy==1.23.5 scipy==1.9.3"
TORCH="torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu"
pip install --no-cache-dir $PIN
pip install --no-cache-dir $PIN $TORCH
pip install --no-cache-dir $PIN $TORCH torch-scatter==2.0.9 torch-sparse==0.6.13 \
  -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install --no-cache-dir $PIN $TORCH torch-geometric==2.0.4 pandas==1.5.3 \
  scikit-learn==1.1.3 matplotlib==3.6.3
python - <<'PY'
import torch
from torch_geometric.nn import TAGConv
out, _ = torch.nn.LSTM(93, 50)(torch.randn(7, 93))
assert out.shape == (7, 50), out.shape
assert len(TAGConv(50, 2, K=3).lins) == 4
print("env ok, torch", torch.__version__)
PY

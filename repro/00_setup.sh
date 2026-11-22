#!/bin/bash
# Python 3.12 environment (.venv) with uv. Tested on Ubuntu (WSL2) + RTX 5080 (CUDA 12.8).
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
# official metric (Royer Lab baseline repository, BSD-3-Clause), used by pipe/score_csv.py
if [ ! -d ref_royerlab ]; then
  git clone https://github.com/royerlab/kaggle-cell-tracking-competition ref_royerlab
  git -C ref_royerlab checkout 075fc5f5a52d11077f9dc2b074644618f26939e2
fi
uv pip install --python .venv/bin/python --no-deps -e ref_royerlab
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

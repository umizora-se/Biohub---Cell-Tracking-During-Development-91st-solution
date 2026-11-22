#!/bin/bash
# Collect the trained weights into build/weights/ and build the three Kaggle notebooks into build/notebooks/.
# build/weights/ has the same file names as weights/ (the weights used for the actual submissions).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/weights
cp det/runs/cnn_all.pt build/weights/
cp pipe/divres/divclf3_s{0,1,2}.txt pipe/divres/divclf3r_s{0,1,2}.txt pipe/divres/divclf_et.npz build/weights/
ls -la build/weights
.venv/bin/python sub/build_notebooks.py --out build/notebooks "$@"

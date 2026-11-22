#!/bin/bash
# 3D-CNN cell-state model (continue / parent / daughter).
# 1) 3-frame GT patches from all 199 training movies -> det/mitosis_patches.npz
# 2) 5 fold models (excluding the movies of fold k, used for honest OOF features) + 1 model on all data (used at inference)
set -euo pipefail
cd "$(dirname "$0")/../det"
PY=../.venv/bin/python
[ -f mitosis_patches.npz ] || nice -n 8 $PY mitosis_data.py > mitosis_data.log 2>&1
mkdir -p runs
for k in 0 1 2 3 4; do
  [ -f runs/cnn_f$k.pt ] || nice -n 8 $PY mitosis_cnn.py train --exclude folds/fold$k.txt --out runs/cnn_f$k.pt > runs/cnn_f$k.log 2>&1
done
[ -f runs/cnn_all.pt ] || nice -n 8 $PY mitosis_cnn.py train --out runs/cnn_all.pt > runs/cnn_all.log 2>&1
tail -n 1 runs/*.log

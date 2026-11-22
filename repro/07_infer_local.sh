#!/bin/bash
# Run one of the three submission notebooks locally (GPU), the same way Kaggle runs it.
# usage: repro/07_infer_local.sh NOTEBOOK [WEIGHTS_DIR] [MOVIE_LIST]
#   NOTEBOOK     final_et07 | second_et05 | best_private_readd_mix
#   WEIGHTS_DIR  default weights/ (submitted weights); use build/weights for your own trained weights
#   MOVIE_LIST   text file with movie names from data/train (default pipe/eval98.txt); they are mounted as the test set
# Output: pipe/exps/infer_NOTEBOOK/kaggle/working/submission.csv  (scored against the training GT with pipe/score_csv.py)
set -euo pipefail
cd "$(dirname "$0")/.."
NB=${1:?notebook name}; W=$(realpath "${2:-weights}"); LIST=$(realpath "${3:-pipe/eval98.txt}")
PY=$(pwd)/.venv/bin/python
[ -f build/notebooks/$NB/notebook.ipynb ] || $PY sub/build_notebooks.py --out build/notebooks
# the notebook cells locate the weights with a glob over the Kaggle input mount -> expose exactly one weights dataset
rm -rf pipe/kaggle/input/datasets/solution && mkdir -p pipe/kaggle/input/datasets/solution
ln -sfn "$W" pipe/kaggle/input/datasets/solution/biohub-divclf3-weights
cd pipe
EXP=infer_$NB
rm -rf exps/$EXP
cmd=$($PY make_exp.py $EXP --nb ../build/notebooks/$NB/notebook.ipynb --eval "$LIST")
nice -n 8 bash -c "$cmd"
nice -n 10 $PY score_csv.py exps/$EXP/kaggle/working/submission.csv > exps/$EXP/score.log 2>&1
grep SCORE exps/$EXP/score.log

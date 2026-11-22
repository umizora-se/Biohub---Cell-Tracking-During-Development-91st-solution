#!/bin/bash
# Run the public base notebook locally on the 98 training movies used for validation (pipe/eval98.txt).
# Output: pipe/exps/r_base98/kaggle/working/{submission.csv, edge_cache/*.npz}
# The edge cache (pre-ILP detections + candidate edges) is reused by the cell-state ILP step (05_readd.sh).
set -euo pipefail
cd "$(dirname "$0")/../pipe"
PY=../.venv/bin/python
EXP=r_base98
if [ ! -f exps/$EXP/kaggle/working/submission.csv ]; then
  rm -rf exps/$EXP
  cmd=$($PY make_exp.py $EXP --eval eval98.txt --set BIOHUB_OUTPUT_MIN_TRACK_LEN=4)
  nice -n 8 bash -c "$cmd"
fi
nice -n 10 $PY score_csv.py exps/$EXP/kaggle/working/submission.csv > exps/$EXP/score.log 2>&1
grep SCORE exps/$EXP/score.log   # original run: SCORE=0.9231

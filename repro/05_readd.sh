#!/bin/bash
# Cell-state ILP + readd (used by best_private_readd_mix).
# OOF CNN probs for all ILP candidates -> ILP re-solve with CNN-adjusted costs -> post-processing with ILP forks
# re-added after motion relink -> classifier retrained on this graph (+ OOF mix with the v3 classifier).
set -euo pipefail
cd "$(dirname "$0")/../pipe"
PY=../.venv/bin/python
nice -n 8 $PY divres/coords_probs.py > divres/coords_probs.log 2>&1                  # divres/ilp_probs/*.npz
[ -d exps/r_ilp2 ] || nice -n 8 $PY ilp_rerun2.py r_base98 r_ilp2 --alpha 0.8 --beta 0 --gamma 1 --rho 0.5 --jobs 3 > divres/ilp2.log 2>&1
# Post-processing with ILP forks re-added. The local run occasionally segfaults on WSL after many movies,
# so the 98 movies are processed in chunks of 20 and a failed chunk is retried.
split -l 20 -d <(tr ' ' '\n' < eval98.txt | grep -v '^$') divres/readd_chunk_
for c in divres/readd_chunk_*; do
  i=${c##*_}; EXP=r_readd_$i; n_movies=$(wc -l < $c)
  for try in 1 2 3 4 5; do
    n=$(grep -c 'FINAL:' exps/$EXP/run.log 2>/dev/null || true)
    [ "${n:-0}" -ge "$n_movies" ] && [ -f exps/$EXP/kaggle/working/submission.csv ] && break
    rm -rf exps/$EXP
    cmd=$($PY make_exp.py $EXP --reuse r_ilp2 --eval $c --replace 5:../sub/replaces/csilp_pp.py)
    nice -n 8 bash -c "$cmd" || echo "chunk $i try $try failed"
  done
  [ -f exps/$EXP/kaggle/working/submission.csv ] || { echo "chunk $i failed"; exit 1; }
done
$PY - <<'PY'
import glob, numpy as np, pandas as pd
d = pd.concat([pd.read_csv(f"exps/r_readd_{c[-2:]}/kaggle/working/submission.csv") for c in sorted(glob.glob("divres/readd_chunk_*"))], ignore_index=True)
d["id"] = np.arange(len(d)); d.to_csv("divres/readd98.csv", index=False); print("readd98 movies", d.dataset.nunique())
PY
nice -n 10 $PY score_csv.py divres/readd98.csv > divres/readd98.score 2>&1; grep SCORE divres/readd98.score   # original: 0.9411
nice -n 8 $PY divres/cv_readd.py divres/readd98.csv divres/oofR > divres/cv_readd.log 2>&1
nice -n 8 $PY divres/final_readd.py                                                  # divres/divclf3r_s{0,1,2}.txt
nice -n 8 $PY divres/v3oof_on_readd.py > divres/v3oof_on_readd.log 2>&1
f=divres/oofR/sub_oof_mix_t0.4_p0.05
nice -n 10 $PY score_csv.py $f.csv > $f.score 2>&1; grep SCORE $f.score                # original: 0.9483

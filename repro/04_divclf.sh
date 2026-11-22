#!/bin/bash
# Division classifier on the base graph (used by final_et07 / second_et05).
# candidates + image features + labels -> temporal features -> CNN patches -> OOF CNN features + grouped 5-fold CV
# -> end-to-end OOF score -> final LightGBM (3 seeds) and ExtraTrees (numpy export).
set -euo pipefail
cd "$(dirname "$0")/../pipe"
PY=../.venv/bin/python
SUB=exps/r_base98/kaggle/working/submission.csv
nice -n 8 $PY divclf_oof.py $SUB divres/oof > divres/oof_v1.log 2>&1                 # divres/oof/cands_all.parquet
[ -f divres/oof/cands_all2.parquet ] || nice -n 8 $PY divres/extend2.py              # + temporal features
nice -n 8 $PY ../det/cand_patches.py divres/oof/cands_all2.parquet $SUB ../det/cand_patches > ../det/cand_patches.log 2>&1
nice -n 8 $PY divres/cv3.py > divres/cv3.log 2>&1                                    # divres/oof3/{cands,forks}_oof3.parquet
grep AUC divres/cv3.log                                                              # original: v2 cands AUC 0.900, v3 cands AUC 0.965
nice -n 10 $PY score_csv.py divres/oof3/sub3_top50_p0.05.csv > divres/oof3/sub3_top50_p0.05.score 2>&1
grep SCORE divres/oof3/sub3_top50_p0.05.score                                        # original: SCORE=0.9396
nice -n 8 $PY divres/final34.py v3                                                   # divres/divclf3_s{0,1,2}.txt
nice -n 8 $PY et_export.py                                                           # divres/divclf_et.npz

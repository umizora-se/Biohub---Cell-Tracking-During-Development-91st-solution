"""Train the final classifier on the readd (cell-state ILP) graph: LightGBM, 3 seeds, all labelled rows of
divres/oofR/table.parquet (orphan-start candidates + existing forks, as in the grouped CV of cv_readd.py).
Output: divres/divclf3r_s{0,1,2}.txt (averaged 50:50 with divclf3_s* in the readd-mix notebook, BIOHUB_DIVCLF_MIX=1).
The original script for this step was not kept; this reconstruction (same features, params and rows as cv_readd.py)
reproduces the submitted weights byte-for-byte.
usage: python divres/final_readd.py   (run from pipe/)
"""
import sys

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, ".")
import divclf3  # noqa: E402
from divclf_oof import LGB_PARAMS  # noqa: E402

T = pd.read_parquet("divres/oofR/table.parquet")
L = T[T.label >= 0]
print("labelled rows", len(L), L.groupby(["kind", "label"]).size().to_dict())
for s in range(3):
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": s}).fit(divclf3.design3(L), L.label)
    m.booster_.save_model(f"divres/divclf3r_s{s}.txt")
print("saved divres/divclf3r_s{0,1,2}.txt")

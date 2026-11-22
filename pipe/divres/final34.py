"""Train final LightGBM ensemble (3 seeds) on all labelled candidates. usage: python divres/final34.py v3|v4"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import divclf3  # noqa: E402
from divclf_oof import LGB_PARAMS  # noqa: E402

variant = sys.argv[1]
if variant == "v3":
    C = pd.read_parquet("divres/oof3/cands_oof3.parquet"); des = divclf3.design3
else:
    C = pd.read_parquet("divres/oof4/cands_oof4.parquet")
    FEATS4 = divclf3.FEATS3 + ["o_linked", "d_o_oldpar"]
    def des(c):
        X = c[FEATS4].astype(float).copy(); X[divclf3.LOG3] = np.log(X[divclf3.LOG3].clip(lower=1e-3)); return X
L = C[C.label >= 0]
print(variant, "labelled", len(L), int(L.label.sum()))
for s in range(3):
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": s}).fit(des(L), L.label)
    m.booster_.save_model(f"divres/divclf3_s{s}.txt")
print("saved divres/divclf3_s{0,1,2}.txt (", variant, ")")

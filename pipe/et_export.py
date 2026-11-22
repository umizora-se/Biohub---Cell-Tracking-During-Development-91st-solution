"""Train the final ExtraTrees division classifier on all labelled base-graph candidates and export it as plain
numpy arrays (portable: no sklearn needed at inference).  Verifies the numpy predictor matches sklearn exactly.
Output: divres/divclf_et.npz
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, ".")
import divclf3  # noqa: E402

C = pd.read_parquet("divres/oof3/cands_oof3.parquet")
L = C[C.label >= 0]
X = divclf3.design3(L).fillna(0).to_numpy(np.float64)
y = L.label.to_numpy()
et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=4, random_state=0).fit(X, y)

feat, thr, left, right, val, offs = [], [], [], [], [], [0]
for est in et.estimators_:
    t = est.tree_
    v = t.value[:, 0, :]
    p1 = v[:, 1] / np.maximum(v.sum(1), 1e-12)
    feat.append(t.feature); thr.append(t.threshold); left.append(t.children_left); right.append(t.children_right); val.append(p1)
    offs.append(offs[-1] + t.node_count)
np.savez_compressed("divres/divclf_et.npz", feature=np.concatenate(feat).astype(np.int32), threshold=np.concatenate(thr),
                    left=np.concatenate(left).astype(np.int32), right=np.concatenate(right).astype(np.int32),
                    value=np.concatenate(val), offsets=np.array(offs, np.int64), n_features=X.shape[1],
                    columns=np.array(divclf3.FEATS3))


def et_predict(npz, X):
    f, th, lf, rt, vl, of = npz["feature"], npz["threshold"], npz["left"], npz["right"], npz["value"], npz["offsets"]
    out = np.zeros(len(X))
    rows = np.arange(len(X))
    for k in range(len(of) - 1):
        base = of[k]
        node = np.zeros(len(X), np.int64)
        while True:
            g = base + node
            leaf = lf[g] == -1
            if leaf.all():
                break
            idx = rows[~leaf]
            gi = g[~leaf]
            go_left = X[idx, f[gi]] <= th[gi]
            node[idx] = np.where(go_left, lf[gi], rt[gi])
        out += vl[base + node]
    return out / (len(of) - 1)


npz = np.load("divres/divclf_et.npz")
A = C.sample(5000, random_state=0)
XA = divclf3.design3(A).fillna(0).to_numpy(np.float64)
ref = et.predict_proba(XA)[:, 1]
mine = et_predict(npz, XA)
print("max abs diff numpy vs sklearn:", float(np.abs(ref - mine).max()))
import os
print("npz size MB:", os.path.getsize("divres/divclf_et.npz") / 1e6)

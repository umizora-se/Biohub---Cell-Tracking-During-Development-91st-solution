"""divclf + temporal intensity features (parent's predecessor, daughters' successors)."""
import numpy as np
import pandas as pd

try:
    from divclf import FEATS as _F1, LOG_FEATS as _L1  # local
except ImportError:  # pasted into the notebook after divclf source
    _F1, _L1 = FEATS, LOG_FEATS  # noqa: F821

EXTRA = ["pp_ctr", "p_rise", "on_ctr", "cn_ctr", "o_drop", "c_drop", "o_fill", "c_fill", "pp_fill"]
FEATS2 = list(_F1) + EXTRA
LOG2 = list(_L1) + ["pp_ctr", "on_ctr", "cn_ctr"]


def _stats_for(nodes, ids, img, q):
    lo, hi = q["0.01"], q["0.999"]
    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]
    sub = P.loc[[i for i in ids if i in P.index]]
    out = {}
    for t, g in sub.groupby("t"):
        v = (np.asarray(img[int(t)], dtype=np.float32) - lo) / (hi - lo)
        for n, r in g.iterrows():
            z, y, x = int(round(r.z)), int(round(r.y)), int(round(r.x))
            w = v[max(z - 1, 0):z + 2, max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]
            bg = v[max(z - 2, 0):z + 3, max(y - 20, 0):y + 21, max(x - 20, 0):x + 21]
            m = float(w.max())
            out[n] = (m / (float(np.median(bg)) + 1e-3), float((w > 0.5 * m).mean()))
    return out


def add_temporal_features(cands, nodes, edges, img, q):
    prv = dict(zip(edges.target_id, edges.source_id))
    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()
    pp = [prv.get(p, p) for p in cands.p]
    on = [nxt[o][0] if len(nxt.get(o, ())) == 1 else o for o in cands.o]
    cn = [nxt[c][0] if len(nxt.get(c, ())) == 1 else c for c in cands.c]
    S = _stats_for(nodes, set(pp) | set(on) | set(cn) | set(cands.o) | set(cands.c), img, q)
    cands["pp_ctr"] = [S[i][0] for i in pp]
    cands["pp_fill"] = [S[i][1] for i in pp]
    cands["on_ctr"] = [S[i][0] for i in on]
    cands["cn_ctr"] = [S[i][0] for i in cn]
    cands["o_fill"] = [S[i][1] for i in cands.o]
    cands["c_fill"] = [S[i][1] for i in cands.c]
    cands["p_rise"] = cands.p_ctr / (cands.pp_ctr + 1e-3)
    cands["o_drop"] = cands.on_ctr / (cands.o_ctr + 1e-3)
    cands["c_drop"] = cands.cn_ctr / (cands.c_ctr + 1e-3)
    return cands


def design2(cands):
    X = cands[FEATS2].astype(float).copy()
    X[LOG2] = np.log(X[LOG2].clip(lower=1e-3))
    return X

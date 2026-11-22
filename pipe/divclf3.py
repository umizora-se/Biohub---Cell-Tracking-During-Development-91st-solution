"""divclf2 + 3D-CNN cell-state probabilities, and a fork table for scoring EXISTING forks with the same model."""
import numpy as np
import pandas as pd

try:
    from divclf2 import FEATS2 as _F2, LOG2 as _L2
    from divclf import _flen, _blen
except ImportError:  # notebook: sources exec'd into one namespace
    _F2, _L2 = FEATS2, LOG2  # noqa: F821

CNN = ["p_par", "p_dau", "pp_par", "o_dau", "o_par", "c_dau", "c_par"]
FEATS3 = list(_F2) + CNN
LOG3 = list(_L2)
SCALE = np.array([1.625, 0.40625, 0.40625])


def add_cnn_features(cands, edges, probs):
    """probs: dict node_id -> (p_continue, p_parent, p_daughter)."""
    prv = dict(zip(edges.target_id, edges.source_id))
    z = (1 / 3, 1 / 3, 1 / 3)
    g = lambda n: probs.get(n, z)
    cands["p_par"] = [g(n)[1] for n in cands.p]
    cands["p_dau"] = [g(n)[2] for n in cands.p]
    cands["pp_par"] = [g(prv.get(n, n))[1] for n in cands.p]
    cands["o_dau"] = [g(n)[2] for n in cands.o]
    cands["o_par"] = [g(n)[1] for n in cands.o]
    cands["c_dau"] = [g(n)[2] for n in cands.c]
    cands["c_par"] = [g(n)[1] for n in cands.c]
    return cands


def design3(cands):
    X = cands[FEATS3].astype(float).copy()
    X[LOG3] = np.log(X[LOG3].clip(lower=1e-3))
    return X


def forks_table(nodes, edges):
    """Existing forks p->(c1,c2) as candidate rows: o = farther child, c = nearer child."""
    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]
    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()
    prv = dict(zip(edges.target_id, edges.source_id))
    XYZ = P[["z", "y", "x"]].to_numpy(float) * SCALE
    idx = {n: i for i, n in enumerate(P.index)}
    rows = []
    for p, ks in nxt.items():
        if len(ks) < 2:
            continue
        pp = XYZ[idx[p]]
        ks = sorted(ks, key=lambda k: np.linalg.norm(XYZ[idx[k]] - pp))
        c, o = ks[0], ks[1]
        pc, po = XYZ[idx[c]], XYZ[idx[o]]
        pv = XYZ[idx[prv[p]]] if p in prv else pp
        rows.append((int(P.at[p, "t"]), p, o, c, np.linalg.norm(pp - po), np.linalg.norm(pp - pc), np.linalg.norm(pc - po),
                     np.linalg.norm((pc + po) / 2 - pp), abs(np.linalg.norm(pp - po) - np.linalg.norm(pp - pc)),
                     float(np.dot(pc - pp, po - pp) / (np.linalg.norm(pc - pp) * np.linalg.norm(po - pp) + 1e-6)),
                     _flen(o, nxt), _flen(c, nxt), _blen(p, prv), np.linalg.norm(pp - pv), P.at[p, "z"]))
    cols = ["t", "p", "o", "c", "d_po", "d_pc", "d_co", "mid_dev", "sym", "cos_opp", "len_o", "len_c", "len_p_back", "p_speed", "z"]
    return pd.DataFrame(rows, columns=cols)

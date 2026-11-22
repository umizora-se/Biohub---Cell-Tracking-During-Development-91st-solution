"""Learned division rescue: orphan track start o near single-child parent p -> add fork p->o if classifier agrees.

Self-contained (numpy/pandas/scipy/zarr/lightgbm) so the same file can be pasted into the Kaggle notebook.
  candidates(nodes, edges)                     -> geometric candidate table
  add_image_features(cands, nodes, img, q)     -> adds intensity/contrast features
  apply(sub_df, image_root, model, thr)        -> sub_df with added division edges
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
FEATS = ["d_po", "d_pc", "d_co", "mid_dev", "sym", "cos_opp", "len_o", "len_c", "len_p_back", "p_speed", "z",
         "p_max", "p_ctr", "p_fill", "o_max", "o_ctr", "c_max", "c_ctr", "oc_ratio", "p_vs_kids"]
LOG_FEATS = ["p_ctr", "o_ctr", "c_ctr"]


def _flen(start, nxt, cap=30):
    n, cur = 1, start
    while n < cap and len(nxt.get(cur, ())) == 1:
        cur = nxt[cur][0]
        n += 1
    return n


def _blen(start, prv, cap=30):
    n, cur = 1, start
    while n < cap and cur in prv:
        cur = prv[cur]
        n += 1
    return n


def candidates(nodes, edges, parent_um=12.0, sister_um=18.0, include_linked=False, steal_old_um=6.0):
    """include_linked: also consider o that already has a parent p' (a 'steal' candidate; p' != p, p' must have 1 child)."""
    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]
    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()
    prv = dict(zip(edges.target_id, edges.source_id))
    XYZ = P[["z", "y", "x"]].to_numpy(float) * SCALE
    idx = {n: i for i, n in enumerate(P.index)}
    T = P.t.to_numpy()
    rows = []
    for t in range(int(T.min()), int(T.max())):
        cur = P.index[T == t]
        par = [p for p in cur if len(nxt.get(p, ())) == 1]
        if include_linked:
            orph = [o for o in P.index[T == t + 1] if (o not in prv) or len(nxt.get(prv[o], ())) == 1]
        else:
            orph = [o for o in P.index[T == t + 1] if o not in prv]
        if not par or not orph:
            continue
        tree = cKDTree(XYZ[[idx[p] for p in par]])
        for o in orph:
            po = XYZ[idx[o]]
            for j in tree.query_ball_point(po, parent_um):
                p = par[j]
                c = nxt[p][0]
                pp, pc = XYZ[idx[p]], XYZ[idx[c]]
                sd = np.linalg.norm(pc - po)
                if sd > sister_um:
                    continue
                if o in prv and prv[o] == p:
                    continue
                if o in prv:  # steal candidate: o's current parent is a near neighbour (<= steal_old_um)
                    if np.linalg.norm(XYZ[idx[prv[o]]] - po) > steal_old_um:
                        continue
                pv = XYZ[idx[prv[p]]] if p in prv else pp
                rows.append((t, p, o, c, np.linalg.norm(pp - po), np.linalg.norm(pp - pc), sd,
                             np.linalg.norm((pc + po) / 2 - pp),
                             abs(np.linalg.norm(pp - po) - np.linalg.norm(pp - pc)),
                             float(np.dot(pc - pp, po - pp) / (np.linalg.norm(pc - pp) * np.linalg.norm(po - pp) + 1e-6)),
                             _flen(o, nxt), _flen(c, nxt), _blen(p, prv), np.linalg.norm(pp - pv), P.at[p, "z"],
                             int(o in prv), np.linalg.norm(XYZ[idx[prv[o]]] - po) if o in prv else 0.0))
    cols = ["t", "p", "o", "c", "d_po", "d_pc", "d_co", "mid_dev", "sym", "cos_opp", "len_o", "len_c", "len_p_back", "p_speed", "z",
            "o_linked", "d_o_oldpar"]
    return pd.DataFrame(rows, columns=cols)


def add_image_features(cands, nodes, img, q):
    """img: array-like (T,Z,Y,X); q: zarr image_statistics quantiles dict."""
    if cands.empty:
        for f in FEATS:
            if f not in cands:
                cands[f] = []
        return cands
    lo, hi = q["0.01"], q["0.999"]
    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]
    need = pd.unique(np.concatenate([cands.p.values, cands.o.values, cands.c.values]))
    sub = P.loc[need]
    stats = {}
    for t, g in sub.groupby("t"):
        v = (np.asarray(img[int(t)], dtype=np.float32) - lo) / (hi - lo)
        for n, r in g.iterrows():
            z, y, x = int(round(r.z)), int(round(r.y)), int(round(r.x))
            w = v[max(z - 1, 0):z + 2, max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]
            bg = v[max(z - 2, 0):z + 3, max(y - 20, 0):y + 21, max(x - 20, 0):x + 21]
            m = float(w.max())
            stats[n] = (m, float(w.mean()), m / (float(np.median(bg)) + 1e-3), float((w > 0.5 * m).mean()))
    S = lambda col, k: np.array([stats[n][k] for n in cands[col].values])
    cands["p_max"], cands["p_mean"], cands["p_ctr"], cands["p_fill"] = S("p", 0), S("p", 1), S("p", 2), S("p", 3)
    cands["o_max"], cands["o_ctr"] = S("o", 0), S("o", 2)
    cands["c_max"], cands["c_ctr"] = S("c", 0), S("c", 2)
    cands["oc_ratio"] = cands.o_max / (cands.c_max + 1e-3)
    cands["p_vs_kids"] = cands.p_max / (0.5 * (cands.o_max + cands.c_max) + 1e-3)
    return cands


def design(cands):
    X = cands[FEATS].astype(float).copy()
    X[LOG_FEATS] = np.log(X[LOG_FEATS].clip(lower=1e-3))
    return X


def select(cands, score, thr):
    """greedy: highest score first, one new child per parent, one parent per orphan."""
    c = cands.assign(score=score)
    c = c[c.score >= thr].sort_values("score", ascending=False)
    used_p, used_o, keep = set(), set(), []
    for r in c.itertuples():
        if r.p in used_p or r.o in used_o:
            continue
        used_p.add(r.p)
        used_o.add(r.o)
        keep.append((int(r.p), int(r.o)))
    return pd.DataFrame(keep, columns=["source_id", "target_id"])


def apply(sub, open_image, model, thr):
    """sub: submission DataFrame; open_image(name) -> (img, quantiles); model: lightgbm Booster/Classifier."""
    parts = [sub.drop(columns=["id"], errors="ignore")]
    total = 0
    for name, grp in sub.groupby("dataset", sort=False):
        nodes = grp[grp.row_type == "node"][["node_id", "t", "z", "y", "x"]]
        edges = grp[grp.row_type == "edge"][["source_id", "target_id"]].astype(int)
        cands = candidates(nodes, edges)
        if cands.empty:
            continue
        img, q = open_image(name)
        cands = add_image_features(cands, nodes, img, q)
        X = design(cands)
        s = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        new = select(cands, s, thr)
        total += len(new)
        parts.append(new.assign(dataset=name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))
        print(f"DIVCLF {name}: {len(cands)} candidates -> +{len(new)} forks", flush=True)
    out = pd.concat(parts, ignore_index=True)[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]
    out.insert(0, "id", np.arange(len(out)))
    print(f"DIVCLF total +{total} forks", flush=True)
    return out

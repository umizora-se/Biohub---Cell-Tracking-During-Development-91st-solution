"""Build a labelled table of division candidates (parent p -> orphan o, with p's existing child c).

label = 1 : p matches a GT division parent d (same t) and o matches one of d's daughters
label = 0 : p matches a GT node with exactly one GT child (annotated, non-dividing) -> a fork there would be an FP
label = -1: p unmatched (metric ignores it)
usage: python div_cands.py sub.csv out.parquet [--parent-um 12] [--sister-um 18]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from io_utils import DATA, SCALE, load_geff  # noqa: E402


def flen(start, nxt, cap=30):
    n, cur = 1, start
    while n < cap and len(nxt.get(cur, [])) == 1:
        cur = nxt[cur][0]; n += 1
    return n


def blen(start, prv, cap=30):
    n, cur = 1, start
    while n < cap and cur in prv:
        cur = prv[cur]; n += 1
    return n


def match(pred, gt):
    p2g = {}
    for t, gs in gt.groupby("t"):
        ps = pred[pred.t == t]
        if not len(ps):
            continue
        d = np.linalg.norm(gs[["z", "y", "x"]].to_numpy(float)[:, None] * SCALE - ps[["z", "y", "x"]].to_numpy(float)[None] * SCALE, axis=-1)
        d[d > 7] = 1e6
        i, j = linear_sum_assignment(d)
        for a, b in zip(i, j):
            if d[a, b] < 1e6:
                p2g[ps.index[b]] = gs.index[a]
    return p2g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub"); ap.add_argument("out")
    ap.add_argument("--parent-um", type=float, default=12.0)
    ap.add_argument("--sister-um", type=float, default=18.0)
    a = ap.parse_args()
    sub = pd.read_csv(a.sub)
    rows = []
    for name, grp in sub.groupby("dataset", sort=False):
        P = grp[grp.row_type == "node"].set_index("node_id")[["t", "z", "y", "x"]]
        E = grp[grp.row_type == "edge"][["source_id", "target_id"]].astype(int)
        nxt = E.groupby("source_id").target_id.apply(list).to_dict()
        prv = dict(zip(E.target_id, E.source_id))
        G, GE, _ = load_geff(DATA / "train" / f"{name}.geff")
        G = G.set_index("id")
        gk = GE.groupby("src").dst.apply(list).to_dict()
        p2g = match(P, G)
        XYZ = P[["z", "y", "x"]].to_numpy(float) * SCALE
        idx = {n: i for i, n in enumerate(P.index)}
        for t in range(int(P.t.min()), int(P.t.max())):
            cur = P.index[(P.t == t).to_numpy()]
            par = [p for p in cur if len(nxt.get(p, [])) == 1]
            orph = [o for o in P.index[(P.t == t + 1).to_numpy()] if o not in prv]
            if not par or not orph:
                continue
            tree = cKDTree(XYZ[[idx[p] for p in par]])
            for o in orph:
                po = XYZ[idx[o]]
                for j in tree.query_ball_point(po, a.parent_um):
                    p = par[j]; c = nxt[p][0]
                    pp, pc = XYZ[idx[p]], XYZ[idx[c]]
                    sd = np.linalg.norm(pc - po)
                    if sd > a.sister_um:
                        continue
                    lab = -1
                    g = p2g.get(p)
                    if g is not None:
                        kids = gk.get(g, [])
                        if len(kids) >= 2:
                            lab = int(p2g.get(o) in kids)
                        elif len(kids) == 1:
                            lab = 0
                    mid = (pc + po) / 2
                    # parent motion before
                    pv = XYZ[idx[prv[p]]] if p in prv else pp
                    rows.append(dict(
                        ds=name, t=t, p=p, o=o, c=c, label=lab,
                        d_po=np.linalg.norm(pp - po), d_pc=np.linalg.norm(pp - pc), d_co=sd,
                        mid_dev=np.linalg.norm(mid - pp),
                        sym=abs(np.linalg.norm(pp - po) - np.linalg.norm(pp - pc)),
                        cos_opp=float(np.dot(pc - pp, po - pp) / (np.linalg.norm(pc - pp) * np.linalg.norm(po - pp) + 1e-6)),
                        len_o=flen(o, nxt), len_c=flen(c, nxt), len_p_back=blen(p, prv),
                        p_speed=np.linalg.norm(pp - pv),
                        n_orph_near=len(tree.query_ball_point(po, 6.0)),
                        z=P.loc[p, "z"], group=name[:4],
                    ))
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    print(df.label.value_counts())


if __name__ == "__main__":
    main()

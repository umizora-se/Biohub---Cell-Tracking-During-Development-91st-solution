"""Retrain the division classifier on candidates generated from the readd (cell-state ILP) output.

Steps (resumable): OOF CNN probs for all nodes -> candidate + fork tables with image/temporal/CNN features and
labels -> grouped 5-fold LightGBM (3 seeds) -> e2e submissions (add thr / prune thr grid) -> write list.
usage: python divres/cv_readd.py SUB.csv OUTDIR
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, "."); sys.path.insert(0, "../det"); sys.path.insert(0, "../eda")
import divclf, divclf2, divclf3  # noqa: E402
from divclf_oof import LGB_PARAMS  # noqa: E402
from div_cands import match  # noqa: E402
from io_utils import DATA, load_geff  # noqa: E402

SUB, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "probs").mkdir(exist_ok=True)
folds = {k: set(Path(f"../det/folds/fold{k}.txt").read_text().split()) for k in range(5)}
ds2fold = {n: k for k, s in folds.items() for n in s}


def step_probs(sub):
    import torch
    from mitosis_cnn import load_model
    from all_node_probs_lib import node_probs
    dev = torch.device("cuda")
    models = {k: load_model(Path(f"../det/runs/cnn_f{k}.pt"), dev) for k in range(5)}
    for name, g in sub.groupby("dataset", sort=True):
        f = OUT / "probs" / f"{name}.parquet"
        if f.exists():
            continue
        nodes = g[g.row_type == "node"].set_index("node_id")[["t", "z", "y", "x"]]
        zp = DATA / "train" / f"{name}.zarr"
        q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
        img = zarr.open_array(str(zp / "0"), mode="r")
        node_probs(nodes, img, q, models[ds2fold[name]], dev).to_parquet(f)
        print("probs", name, flush=True)


def build(name):
    sub = pd.read_csv(SUB)
    g = sub[sub.dataset == name]
    nodes = g[g.row_type == "node"][["node_id", "t", "z", "y", "x"]]
    edges = g[g.row_type == "edge"][["source_id", "target_id"]].astype(int)
    pr = pd.read_parquet(OUT / "probs" / f"{name}.parquet").set_index("node_id")
    probs = {n: (a, b, c) for n, a, b, c in zip(pr.index, pr.p_cont, pr.p_par, pr.p_dau)}
    zp = DATA / "train" / f"{name}.zarr"
    q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
    img = zarr.open_array(str(zp / "0"), mode="r")
    G, GE, _ = load_geff(DATA / "train" / f"{name}.geff"); G = G.set_index("id")
    gk = GE.groupby("src").dst.apply(list).to_dict()
    p2g = match(nodes.set_index("node_id")[["t", "z", "y", "x"]], G)
    out = []
    for kind, c in (("add", divclf.candidates(nodes, edges)), ("fork", divclf3.forks_table(nodes, edges))):
        if c.empty:
            continue
        c = divclf.add_image_features(c, nodes, img, q)
        c = divclf2.add_temporal_features(c, nodes, edges, img, q)
        c = divclf3.add_cnn_features(c, edges, probs)
        lab = []
        for p, o in zip(c.p, c.o):
            gid = p2g.get(p)
            if gid is None:
                lab.append(-1); continue
            kids = gk.get(gid, [])
            if kind == "add":
                lab.append(int(p2g.get(o) in kids) if len(kids) >= 2 else (0 if len(kids) == 1 else -1))
            else:
                lab.append(1 if len(kids) >= 2 else (0 if len(kids) == 1 else -1))
        c["label"] = lab; c["ds"] = name; c["kind"] = kind
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    sub = pd.read_csv(SUB)
    step_probs(sub)
    tab = OUT / "table.parquet"
    if not tab.exists():
        with ProcessPoolExecutor(3) as ex:
            T = pd.concat([t for t in ex.map(build, sorted(sub.dataset.unique())) if len(t)], ignore_index=True)
        T.to_parquet(tab)
    T = pd.read_parquet(tab)
    print("table", len(T), T.groupby(["kind", "label"]).size().to_dict(), flush=True)
    # grouped CV: one model on add+fork labelled rows (same as v3 design), OOF for all rows
    T["oof"] = 0.0
    for seed in range(3):
        for k in range(5):
            trm = (~T.ds.isin(folds[k]) & (T.label >= 0)).values
            m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed}).fit(divclf3.design3(T[trm]), T.label[trm])
            tem = T.ds.isin(folds[k]).values
            T.loc[tem, "oof"] += m.predict_proba(divclf3.design3(T[tem]))[:, 1] / 3
    T.to_parquet(OUT / "table_oof.parquet")
    for thr, pthr in [(0.4, 0.05), (0.4, 0.1), (0.3, 0.05), (0.5, 0.05), (0.4, 0.2)]:
        parts = []
        for name, grp in sub.groupby("dataset", sort=False):
            nodes = grp[grp.row_type == "node"]; edges = grp[grp.row_type == "edge"][["source_id", "target_id"]].astype(int)
            f = T[(T.ds == name) & (T.kind == "fork") & (T.oof < pthr)]
            rm = set(zip(f.p, f.o))
            edges = edges[[(s, t) not in rm for s, t in zip(edges.source_id, edges.target_id)]]
            a = T[(T.ds == name) & (T.kind == "add")]
            new = divclf.select(a, a.oof.values, thr) if len(a) else pd.DataFrame(columns=["source_id", "target_id"])
            parts.append(nodes.drop(columns=["id"]))
            parts.append(pd.concat([edges, new]).assign(dataset=name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))
        res = pd.concat(parts, ignore_index=True)[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]
        res.insert(0, "id", np.arange(len(res)))
        res.to_csv(OUT / f"sub_t{thr}_p{pthr}.csv", index=False)
        print(f"wrote sub_t{thr}_p{pthr}.csv", flush=True)


if __name__ == "__main__":
    main()

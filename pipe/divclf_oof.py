"""OOF end-to-end check of divclf on the 98-dataset in-sample predictions.

1) all candidates + features + labels  2) GroupKFold(5) LightGBM on labelled rows, predict all rows of held-out datasets
3) for several thresholds write submission with added forks -> score with score_csv
usage: python divclf_oof.py SUB.csv OUTDIR
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import zarr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import divclf  # noqa: E402
from div_cands import match  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from io_utils import DATA, load_geff  # noqa: E402

LGB_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=20, verbose=-1)


def build(args):
    name, grp = args
    nodes = grp[grp.row_type == "node"][["node_id", "t", "z", "y", "x"]]
    edges = grp[grp.row_type == "edge"][["source_id", "target_id"]].astype(int)
    c = divclf.candidates(nodes, edges)
    if c.empty:
        return c
    zp = DATA / "train" / f"{name}.zarr"
    q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
    c = divclf.add_image_features(c, nodes, zarr.open_array(str(zp / "0"), mode="r"), q)
    G, GE, _ = load_geff(DATA / "train" / f"{name}.geff")
    G = G.set_index("id")
    gk = GE.groupby("src").dst.apply(list).to_dict()
    p2g = match(nodes.set_index("node_id")[["t", "z", "y", "x"]], G)
    lab = []
    for p, o in zip(c.p, c.o):
        g = p2g.get(p)
        if g is None:
            lab.append(-1)
        else:
            kids = gk.get(g, [])
            lab.append(int(p2g.get(o) in kids) if len(kids) >= 2 else (0 if len(kids) == 1 else -1))
    c["label"] = lab
    c["ds"] = name
    return c


def main(sub_path, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(sub_path)
    tab_path = out / "cands_all.parquet"
    if tab_path.exists():
        C = pd.read_parquet(tab_path)
    else:
        with ProcessPoolExecutor(4) as ex:
            C = pd.concat(list(ex.map(build, list(sub.groupby("dataset", sort=False)))), ignore_index=True)
        C.to_parquet(tab_path)
    print("candidates", len(C), C.label.value_counts().to_dict(), flush=True)
    C["oof"] = np.nan
    names = np.array(sorted(C.ds.unique()))
    for k, (tr, te) in enumerate(GroupKFold(5).split(names, groups=names)):
        trm = C.ds.isin(names[tr]) & (C.label >= 0)
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(divclf.design(C[trm]), C.label[trm])
        tem = C.ds.isin(names[te])
        C.loc[tem, "oof"] = m.predict_proba(divclf.design(C[tem]))[:, 1]
    C.to_parquet(out / "cands_oof.parquet")
    L = C[C.label >= 0].sort_values("oof", ascending=False)
    for k in (20, 40, 80, 150):
        thr = L.oof.iloc[k - 1]
        print(f"top{k} labelled -> thr={thr:.4f}; all candidates above thr: {(C.oof >= thr).sum()}", flush=True)
    for k in (20, 40, 80, 150):
        thr = float(L.oof.iloc[k - 1])
        parts = [sub.drop(columns=["id"])]
        for name, c in C.groupby("ds"):
            new = divclf.select(c, c.oof.values, thr)
            parts.append(new.assign(dataset=name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))
        res = pd.concat(parts, ignore_index=True)[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]
        res.insert(0, "id", np.arange(len(res)))
        res.to_csv(out / f"sub_top{k}.csv", index=False)
        print(f"wrote sub_top{k}.csv (+{len(res) - len(sub)} forks)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])

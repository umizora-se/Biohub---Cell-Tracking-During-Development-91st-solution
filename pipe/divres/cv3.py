"""divclf3 OOF: CNN fold models -> node probs -> features; LightGBM grouped CV; end-to-end submissions.
Also scores existing forks (fork table) with the same OOF models for pruning.
usage: python divres/cv3.py            (run from pipe/)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import zarr

sys.path.insert(0, "."); sys.path.insert(0, "../det"); sys.path.insert(0, "../eda")
import divclf, divclf2, divclf3  # noqa: E402
from divclf_oof import LGB_PARAMS  # noqa: E402
from mitosis_cnn import load_model, predict  # noqa: E402
from div_cands import match  # noqa: E402
from io_utils import DATA, load_geff  # noqa: E402

DET = Path("../det")
OUT = Path("divres/oof3"); OUT.mkdir(exist_ok=True)
SUB = "exps/r_base98/kaggle/working/submission.csv"
dev = torch.device("cuda")

C = pd.read_parquet("divres/oof/cands_all2.parquet")
sub = pd.read_csv(SUB)
names = np.array(sorted(C.ds.unique()))
folds = {k: set(Path(f"../det/folds/fold{k}.txt").read_text().split()) for k in range(5)}
ds2fold = {n: k for k, s in folds.items() for n in s}

# 1) CNN OOF node probabilities
models = {k: load_model(DET / f"runs/cnn_f{k}.pt", dev) for k in range(5)}
probs_all = {}
for name in names:
    z = np.load(DET / f"cand_patches/{name}.npz")
    pr = predict(models[ds2fold[name]], z["X"], dev)
    probs_all[name] = dict(zip(z["ids"].tolist(), map(tuple, pr)))
print("cnn probs done", flush=True)

# 2) features for candidates + fork table (with labels)
parts, fparts = [], []
for name, c in C.groupby("ds"):
    g = sub[sub.dataset == name]
    nodes = g[g.row_type == "node"][["node_id", "t", "z", "y", "x"]]
    edges = g[g.row_type == "edge"][["source_id", "target_id"]].astype(int)
    parts.append(divclf3.add_cnn_features(c.copy(), edges, probs_all[name]))
    ft = divclf3.forks_table(nodes, edges)
    if len(ft):
        zp = DATA / "train" / f"{name}.zarr"
        q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
        img = zarr.open_array(str(zp / "0"), mode="r")
        ft = divclf.add_image_features(ft, nodes, img, q)
        ft = divclf2.add_temporal_features(ft, nodes, edges, img, q)
        # CNN probs for fork nodes may be missing (not in candidate set) -> extract on the fly
        need = [n for n in set(ft.p) | set(ft.o) | set(ft.c) if n for n in [n] if n not in probs_all[name]]
        if need:
            from mitosis_data import extract
            P = nodes.set_index("node_id")
            X = np.stack([extract(img, q, int(P.at[n, "t"]), int(P.at[n, "z"]), int(P.at[n, "y"]), int(P.at[n, "x"])) for n in need])
            pr = predict(models[ds2fold[name]], X, dev)
            probs_all[name].update(dict(zip(need, map(tuple, pr))))
        ft = divclf3.add_cnn_features(ft, edges, probs_all[name])
        # labels: parent matched to GT node with >=2 kids -> 1, with 1 kid -> 0, else -1
        G, GE, _ = load_geff(DATA / "train" / f"{name}.geff"); G = G.set_index("id")
        gk = GE.groupby("src").dst.apply(list).to_dict()
        p2g = match(nodes.set_index("node_id")[["t", "z", "y", "x"]], G)
        lab = []
        for p in ft.p:
            gid = p2g.get(p)
            lab.append(-1 if gid is None else (1 if len(gk.get(gid, [])) >= 2 else (0 if len(gk.get(gid, [])) == 1 else -1)))
        ft["label"] = lab; ft["ds"] = name
        fparts.append(ft)
C3 = pd.concat(parts, ignore_index=True)
FT = pd.concat(fparts, ignore_index=True)
print("candidates", len(C3), "forks", len(FT), FT.label.value_counts().to_dict(), flush=True)

# 3) grouped CV LightGBM (3 seeds), OOF on candidates and on forks
from sklearn.metrics import roc_auc_score, average_precision_score
for tag, des in [("v2", divclf2.design2), ("v3", divclf3.design3)]:
    C3["oof_" + tag] = 0.0; FT["oof_" + tag] = 0.0
    for seed in range(3):
        for k in range(5):
            trm = ~C3.ds.isin(folds[k]) & (C3.label >= 0)
            m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed}).fit(des(C3[trm]), C3.label[trm])
            tem = C3.ds.isin(folds[k]); C3.loc[tem, "oof_" + tag] += m.predict_proba(des(C3[tem]))[:, 1] / 3
            ftm = FT.ds.isin(folds[k]); FT.loc[ftm, "oof_" + tag] += m.predict_proba(des(FT[ftm]))[:, 1] / 3
    L = C3[C3.label >= 0]; y = L.label.values; o = L["oof_" + tag].values; s = np.argsort(-o)
    print(tag, "cands AUC %.3f AP %.3f" % (roc_auc_score(y, o), average_precision_score(y, o)), {k: int(y[s[:k]].sum()) for k in [20, 40, 80]})
    Lf = FT[FT.label >= 0]
    print(tag, "forks AUC %.3f" % roc_auc_score(Lf.label, Lf["oof_" + tag]), "n", len(Lf), flush=True)
C3.to_parquet(OUT / "cands_oof3.parquet"); FT.to_parquet(OUT / "forks_oof3.parquet")

# 4) end-to-end submissions: add forks (thr from top-k labelled) and optionally prune existing forks below pthr
L = C3[C3.label >= 0].sort_values("oof_v3", ascending=False)
for k in (30, 50):
    thr = float(L.oof_v3.iloc[k - 1])
    for pthr in (0.0, 0.05, 0.1):
        parts = []
        for name, grp in sub.groupby("dataset", sort=False):
            nodes = grp[grp.row_type == "node"]; edges = grp[grp.row_type == "edge"][["source_id", "target_id"]].astype(int)
            if pthr > 0:
                bad = FT[(FT.ds == name) & (FT.oof_v3 < pthr)]
                rm = set(zip(bad.p, bad.o))
                edges = edges[[(s, t) not in rm for s, t in zip(edges.source_id, edges.target_id)]]
            c = C3[C3.ds == name]
            new = divclf.select(c, c.oof_v3.values, thr)
            parts.append(nodes.drop(columns=["id"]))
            parts.append(pd.concat([edges, new]).assign(dataset=name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))
        res = pd.concat(parts, ignore_index=True)[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]
        res.insert(0, "id", np.arange(len(res)))
        res.to_csv(OUT / f"sub3_top{k}_p{pthr}.csv", index=False)
        print(f"wrote sub3_top{k}_p{pthr}.csv thr={thr:.3f} rows={len(res)}", flush=True)

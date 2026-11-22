# ==== divclf3: learned division rescue + fork pruning
# 3D-CNN cell-state classifier (continue/parent/daughter, trained offline on GT tracks, cf. Hirsch et al. 2022)
# + LightGBM on geometry / intensity / temporal / CNN features. Adds forks p->o for orphan track starts next to a
# single-child parent; removes existing forks the model rejects. Weights from attached dataset biohub-divclf3-weights.
import traceback as _dc_tb
DIVCLF_THRESHOLD = float(os.environ.get("BIOHUB_DIVCLF_THRESHOLD", "0.4"))
DIVCLF_PRUNE_THRESHOLD = float(os.environ.get("BIOHUB_DIVCLF_PRUNE_THRESHOLD", "0.05"))
DIVCLF_MAX_ADD = int(os.environ.get("BIOHUB_DIVCLF_MAX_ADD", "0"))  # per movie; 0 = no cap
_SRC_DIVCLF = '"""Learned division rescue: orphan track start o near single-child parent p -> add fork p->o if classifier agrees.\n\nSelf-contained (numpy/pandas/scipy/zarr/lightgbm) so the same file can be pasted into the Kaggle notebook.\n  candidates(nodes, edges)                     -> geometric candidate table\n  add_image_features(cands, nodes, img, q)     -> adds intensity/contrast features\n  apply(sub_df, image_root, model, thr)        -> sub_df with added division edges\n"""\nimport numpy as np\nimport pandas as pd\nfrom scipy.spatial import cKDTree\n\nSCALE = np.array([1.625, 0.40625, 0.40625])\nFEATS = ["d_po", "d_pc", "d_co", "mid_dev", "sym", "cos_opp", "len_o", "len_c", "len_p_back", "p_speed", "z",\n         "p_max", "p_ctr", "p_fill", "o_max", "o_ctr", "c_max", "c_ctr", "oc_ratio", "p_vs_kids"]\nLOG_FEATS = ["p_ctr", "o_ctr", "c_ctr"]\n\n\ndef _flen(start, nxt, cap=30):\n    n, cur = 1, start\n    while n < cap and len(nxt.get(cur, ())) == 1:\n        cur = nxt[cur][0]\n        n += 1\n    return n\n\n\ndef _blen(start, prv, cap=30):\n    n, cur = 1, start\n    while n < cap and cur in prv:\n        cur = prv[cur]\n        n += 1\n    return n\n\n\ndef candidates(nodes, edges, parent_um=12.0, sister_um=18.0, include_linked=False, steal_old_um=6.0):\n    """include_linked: also consider o that already has a parent p\' (a \'steal\' candidate; p\' != p, p\' must have 1 child)."""\n    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]\n    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()\n    prv = dict(zip(edges.target_id, edges.source_id))\n    XYZ = P[["z", "y", "x"]].to_numpy(float) * SCALE\n    idx = {n: i for i, n in enumerate(P.index)}\n    T = P.t.to_numpy()\n    rows = []\n    for t in range(int(T.min()), int(T.max())):\n        cur = P.index[T == t]\n        par = [p for p in cur if len(nxt.get(p, ())) == 1]\n        if include_linked:\n            orph = [o for o in P.index[T == t + 1] if (o not in prv) or len(nxt.get(prv[o], ())) == 1]\n        else:\n            orph = [o for o in P.index[T == t + 1] if o not in prv]\n        if not par or not orph:\n            continue\n        tree = cKDTree(XYZ[[idx[p] for p in par]])\n        for o in orph:\n            po = XYZ[idx[o]]\n            for j in tree.query_ball_point(po, parent_um):\n                p = par[j]\n                c = nxt[p][0]\n                pp, pc = XYZ[idx[p]], XYZ[idx[c]]\n                sd = np.linalg.norm(pc - po)\n                if sd > sister_um:\n                    continue\n                if o in prv and prv[o] == p:\n                    continue\n                if o in prv:  # steal candidate: o\'s current parent is a near neighbour (<= steal_old_um)\n                    if np.linalg.norm(XYZ[idx[prv[o]]] - po) > steal_old_um:\n                        continue\n                pv = XYZ[idx[prv[p]]] if p in prv else pp\n                rows.append((t, p, o, c, np.linalg.norm(pp - po), np.linalg.norm(pp - pc), sd,\n                             np.linalg.norm((pc + po) / 2 - pp),\n                             abs(np.linalg.norm(pp - po) - np.linalg.norm(pp - pc)),\n                             float(np.dot(pc - pp, po - pp) / (np.linalg.norm(pc - pp) * np.linalg.norm(po - pp) + 1e-6)),\n                             _flen(o, nxt), _flen(c, nxt), _blen(p, prv), np.linalg.norm(pp - pv), P.at[p, "z"],\n                             int(o in prv), np.linalg.norm(XYZ[idx[prv[o]]] - po) if o in prv else 0.0))\n    cols = ["t", "p", "o", "c", "d_po", "d_pc", "d_co", "mid_dev", "sym", "cos_opp", "len_o", "len_c", "len_p_back", "p_speed", "z",\n            "o_linked", "d_o_oldpar"]\n    return pd.DataFrame(rows, columns=cols)\n\n\ndef add_image_features(cands, nodes, img, q):\n    """img: array-like (T,Z,Y,X); q: zarr image_statistics quantiles dict."""\n    if cands.empty:\n        for f in FEATS:\n            if f not in cands:\n                cands[f] = []\n        return cands\n    lo, hi = q["0.01"], q["0.999"]\n    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]\n    need = pd.unique(np.concatenate([cands.p.values, cands.o.values, cands.c.values]))\n    sub = P.loc[need]\n    stats = {}\n    for t, g in sub.groupby("t"):\n        v = (np.asarray(img[int(t)], dtype=np.float32) - lo) / (hi - lo)\n        for n, r in g.iterrows():\n            z, y, x = int(round(r.z)), int(round(r.y)), int(round(r.x))\n            w = v[max(z - 1, 0):z + 2, max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]\n            bg = v[max(z - 2, 0):z + 3, max(y - 20, 0):y + 21, max(x - 20, 0):x + 21]\n            m = float(w.max())\n            stats[n] = (m, float(w.mean()), m / (float(np.median(bg)) + 1e-3), float((w > 0.5 * m).mean()))\n    S = lambda col, k: np.array([stats[n][k] for n in cands[col].values])\n    cands["p_max"], cands["p_mean"], cands["p_ctr"], cands["p_fill"] = S("p", 0), S("p", 1), S("p", 2), S("p", 3)\n    cands["o_max"], cands["o_ctr"] = S("o", 0), S("o", 2)\n    cands["c_max"], cands["c_ctr"] = S("c", 0), S("c", 2)\n    cands["oc_ratio"] = cands.o_max / (cands.c_max + 1e-3)\n    cands["p_vs_kids"] = cands.p_max / (0.5 * (cands.o_max + cands.c_max) + 1e-3)\n    return cands\n\n\ndef design(cands):\n    X = cands[FEATS].astype(float).copy()\n    X[LOG_FEATS] = np.log(X[LOG_FEATS].clip(lower=1e-3))\n    return X\n\n\ndef select(cands, score, thr):\n    """greedy: highest score first, one new child per parent, one parent per orphan."""\n    c = cands.assign(score=score)\n    c = c[c.score >= thr].sort_values("score", ascending=False)\n    used_p, used_o, keep = set(), set(), []\n    for r in c.itertuples():\n        if r.p in used_p or r.o in used_o:\n            continue\n        used_p.add(r.p)\n        used_o.add(r.o)\n        keep.append((int(r.p), int(r.o)))\n    return pd.DataFrame(keep, columns=["source_id", "target_id"])\n\n\ndef apply(sub, open_image, model, thr):\n    """sub: submission DataFrame; open_image(name) -> (img, quantiles); model: lightgbm Booster/Classifier."""\n    parts = [sub.drop(columns=["id"], errors="ignore")]\n    total = 0\n    for name, grp in sub.groupby("dataset", sort=False):\n        nodes = grp[grp.row_type == "node"][["node_id", "t", "z", "y", "x"]]\n        edges = grp[grp.row_type == "edge"][["source_id", "target_id"]].astype(int)\n        cands = candidates(nodes, edges)\n        if cands.empty:\n            continue\n        img, q = open_image(name)\n        cands = add_image_features(cands, nodes, img, q)\n        X = design(cands)\n        s = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)\n        new = select(cands, s, thr)\n        total += len(new)\n        parts.append(new.assign(dataset=name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))\n        print(f"DIVCLF {name}: {len(cands)} candidates -> +{len(new)} forks", flush=True)\n    out = pd.concat(parts, ignore_index=True)[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]\n    out.insert(0, "id", np.arange(len(out)))\n    print(f"DIVCLF total +{total} forks", flush=True)\n    return out\n'
_SRC_DIVCLF2 = '"""divclf + temporal intensity features (parent\'s predecessor, daughters\' successors)."""\nimport numpy as np\nimport pandas as pd\n\ntry:\n    from divclf import FEATS as _F1, LOG_FEATS as _L1  # local\nexcept ImportError:  # pasted into the notebook after divclf source\n    _F1, _L1 = FEATS, LOG_FEATS  # noqa: F821\n\nEXTRA = ["pp_ctr", "p_rise", "on_ctr", "cn_ctr", "o_drop", "c_drop", "o_fill", "c_fill", "pp_fill"]\nFEATS2 = list(_F1) + EXTRA\nLOG2 = list(_L1) + ["pp_ctr", "on_ctr", "cn_ctr"]\n\n\ndef _stats_for(nodes, ids, img, q):\n    lo, hi = q["0.01"], q["0.999"]\n    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]\n    sub = P.loc[[i for i in ids if i in P.index]]\n    out = {}\n    for t, g in sub.groupby("t"):\n        v = (np.asarray(img[int(t)], dtype=np.float32) - lo) / (hi - lo)\n        for n, r in g.iterrows():\n            z, y, x = int(round(r.z)), int(round(r.y)), int(round(r.x))\n            w = v[max(z - 1, 0):z + 2, max(y - 6, 0):y + 7, max(x - 6, 0):x + 7]\n            bg = v[max(z - 2, 0):z + 3, max(y - 20, 0):y + 21, max(x - 20, 0):x + 21]\n            m = float(w.max())\n            out[n] = (m / (float(np.median(bg)) + 1e-3), float((w > 0.5 * m).mean()))\n    return out\n\n\ndef add_temporal_features(cands, nodes, edges, img, q):\n    prv = dict(zip(edges.target_id, edges.source_id))\n    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()\n    pp = [prv.get(p, p) for p in cands.p]\n    on = [nxt[o][0] if len(nxt.get(o, ())) == 1 else o for o in cands.o]\n    cn = [nxt[c][0] if len(nxt.get(c, ())) == 1 else c for c in cands.c]\n    S = _stats_for(nodes, set(pp) | set(on) | set(cn) | set(cands.o) | set(cands.c), img, q)\n    cands["pp_ctr"] = [S[i][0] for i in pp]\n    cands["pp_fill"] = [S[i][1] for i in pp]\n    cands["on_ctr"] = [S[i][0] for i in on]\n    cands["cn_ctr"] = [S[i][0] for i in cn]\n    cands["o_fill"] = [S[i][1] for i in cands.o]\n    cands["c_fill"] = [S[i][1] for i in cands.c]\n    cands["p_rise"] = cands.p_ctr / (cands.pp_ctr + 1e-3)\n    cands["o_drop"] = cands.on_ctr / (cands.o_ctr + 1e-3)\n    cands["c_drop"] = cands.cn_ctr / (cands.c_ctr + 1e-3)\n    return cands\n\n\ndef design2(cands):\n    X = cands[FEATS2].astype(float).copy()\n    X[LOG2] = np.log(X[LOG2].clip(lower=1e-3))\n    return X\n'
_SRC_DIVCLF3 = '"""divclf2 + 3D-CNN cell-state probabilities, and a fork table for scoring EXISTING forks with the same model."""\nimport numpy as np\nimport pandas as pd\n\ntry:\n    from divclf2 import FEATS2 as _F2, LOG2 as _L2\n    from divclf import _flen, _blen\nexcept ImportError:  # notebook: sources exec\'d into one namespace\n    _F2, _L2 = FEATS2, LOG2  # noqa: F821\n\nCNN = ["p_par", "p_dau", "pp_par", "o_dau", "o_par", "c_dau", "c_par"]\nFEATS3 = list(_F2) + CNN\nLOG3 = list(_L2)\nSCALE = np.array([1.625, 0.40625, 0.40625])\n\n\ndef add_cnn_features(cands, edges, probs):\n    """probs: dict node_id -> (p_continue, p_parent, p_daughter)."""\n    prv = dict(zip(edges.target_id, edges.source_id))\n    z = (1 / 3, 1 / 3, 1 / 3)\n    g = lambda n: probs.get(n, z)\n    cands["p_par"] = [g(n)[1] for n in cands.p]\n    cands["p_dau"] = [g(n)[2] for n in cands.p]\n    cands["pp_par"] = [g(prv.get(n, n))[1] for n in cands.p]\n    cands["o_dau"] = [g(n)[2] for n in cands.o]\n    cands["o_par"] = [g(n)[1] for n in cands.o]\n    cands["c_dau"] = [g(n)[2] for n in cands.c]\n    cands["c_par"] = [g(n)[1] for n in cands.c]\n    return cands\n\n\ndef design3(cands):\n    X = cands[FEATS3].astype(float).copy()\n    X[LOG3] = np.log(X[LOG3].clip(lower=1e-3))\n    return X\n\n\ndef forks_table(nodes, edges):\n    """Existing forks p->(c1,c2) as candidate rows: o = farther child, c = nearer child."""\n    P = nodes.set_index("node_id")[["t", "z", "y", "x"]]\n    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()\n    prv = dict(zip(edges.target_id, edges.source_id))\n    XYZ = P[["z", "y", "x"]].to_numpy(float) * SCALE\n    idx = {n: i for i, n in enumerate(P.index)}\n    rows = []\n    for p, ks in nxt.items():\n        if len(ks) < 2:\n            continue\n        pp = XYZ[idx[p]]\n        ks = sorted(ks, key=lambda k: np.linalg.norm(XYZ[idx[k]] - pp))\n        c, o = ks[0], ks[1]\n        pc, po = XYZ[idx[c]], XYZ[idx[o]]\n        pv = XYZ[idx[prv[p]]] if p in prv else pp\n        rows.append((int(P.at[p, "t"]), p, o, c, np.linalg.norm(pp - po), np.linalg.norm(pp - pc), np.linalg.norm(pc - po),\n                     np.linalg.norm((pc + po) / 2 - pp), abs(np.linalg.norm(pp - po) - np.linalg.norm(pp - pc)),\n                     float(np.dot(pc - pp, po - pp) / (np.linalg.norm(pc - pp) * np.linalg.norm(po - pp) + 1e-6)),\n                     _flen(o, nxt), _flen(c, nxt), _blen(p, prv), np.linalg.norm(pp - pv), P.at[p, "z"]))\n    cols = ["t", "p", "o", "c", "d_po", "d_pc", "d_co", "mid_dev", "sym", "cos_opp", "len_o", "len_c", "len_p_back", "p_speed", "z"]\n    return pd.DataFrame(rows, columns=cols)\n'
_SRC_CNN = '"""3D cell-state classifier (continue / parent / daughter) on 3-frame patches. Small ResNet, self-contained\n(this file is also embedded in the Kaggle notebook for inference).\n\ntrain:  python mitosis_cnn.py train --fold K --exclude names.txt --out runs/cnn_fK.pt   (K=-1 -> all data)\n"""\nimport argparse\nimport json\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nPZ, PY, PX = 7, 32, 32\n\n\nclass Block(nn.Module):\n    def __init__(self, cin, cout, stride=(1, 2, 2)):\n        super().__init__()\n        self.c1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)\n        self.b1 = nn.BatchNorm3d(cout)\n        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)\n        self.b2 = nn.BatchNorm3d(cout)\n        self.sc = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm3d(cout)) if (cin != cout or stride != (1, 1, 1)) else nn.Identity()\n\n    def forward(self, x):\n        h = F.relu(self.b1(self.c1(x)))\n        h = self.b2(self.c2(h))\n        return F.relu(h + self.sc(x))\n\n\nclass MitosisNet(nn.Module):\n    def __init__(self, in_ch=3, width=24, n_cls=3):\n        super().__init__()\n        self.stem = nn.Sequential(nn.Conv3d(in_ch, width, 3, padding=1, bias=False), nn.BatchNorm3d(width), nn.ReLU())\n        self.l1 = Block(width, width, (1, 1, 1))\n        self.l2 = Block(width, width * 2, (1, 2, 2))\n        self.l3 = Block(width * 2, width * 4, (2, 2, 2))\n        self.l4 = Block(width * 4, width * 4, (1, 2, 2))\n        self.head = nn.Linear(width * 4, n_cls)\n\n    def forward(self, x):\n        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))\n        return self.head(x.mean((2, 3, 4)))\n\n\ndef augment(x, rng):\n    # x: (B,3,PZ,PY,PX) tensor on device\n    if rng.random() < 0.5:\n        x = x.flip(-1)\n    if rng.random() < 0.5:\n        x = x.flip(-2)\n    if rng.random() < 0.5:\n        x = x.transpose(-1, -2)\n    if rng.random() < 0.5:\n        x = x.flip(-3)\n    scale = torch.tensor(rng.uniform(0.8, 1.25, size=(x.shape[0], 1, 1, 1, 1)), dtype=x.dtype, device=x.device)\n    return x * scale\n\n\n@torch.no_grad()\ndef predict(model, X, device, bs=256, tta=True):\n    model.eval()\n    out = []\n    for i in range(0, len(X), bs):\n        xb = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=device)\n        logits = model(xb)\n        if tta:\n            logits = logits + model(xb.flip(-1)) + model(xb.flip(-2)) + model(xb.transpose(-1, -2))\n            logits = logits / 4\n        out.append(F.softmax(logits, 1).cpu().numpy())\n    return np.concatenate(out)\n\n\ndef train(a):\n    d = np.load(Path(__file__).parent / "mitosis_patches.npz")\n    X, y, soft, ds = d["X"], d["y"], d["soft"], d["ds"]\n    excl = set(Path(a.exclude).read_text().split()) if a.exclude else set()\n    m = ~np.isin(ds, list(excl))\n    X, y, soft = X[m], y[m], soft[m]\n    print("train patches", len(y), np.bincount(y), "excluded ds", len(excl), flush=True)\n    device = torch.device("cuda")\n    torch.manual_seed(a.seed)\n    rng = np.random.default_rng(a.seed)\n    model = MitosisNet().to(device)\n    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * (len(y) // a.bs + 1))\n    cw = torch.tensor([1.0, 4.0, 3.0], device=device)\n    w_soft = torch.tensor(np.where(soft == 1, 0.5, 1.0), dtype=torch.float32)\n    # oversample positives\n    p = np.where(y == 0, 1.0, 6.0); p /= p.sum()\n    for ep in range(a.epochs):\n        model.train()\n        idx = rng.choice(len(y), size=len(y), p=p)\n        tot = 0\n        for i in range(0, len(idx), a.bs):\n            b = idx[i:i + a.bs]\n            xb = torch.as_tensor(X[b].astype(np.float32), device=device)\n            xb = augment(xb, rng)\n            yb = torch.as_tensor(y[b], device=device)\n            loss = (F.cross_entropy(model(xb), yb, weight=cw, reduction="none") * w_soft[b].to(device)).mean()\n            opt.zero_grad(); loss.backward(); opt.step(); sched.step()\n            tot += loss.item() * len(b)\n        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(idx):.4f}", flush=True)\n    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)\n    torch.save({"state_dict": {k: v.half() for k, v in model.state_dict().items()}, "arch": "MitosisNet", "width": 24}, out)\n    print("saved", out)\n\n\ndef load_model(path, device):\n    ck = torch.load(path, map_location=device)\n    m = MitosisNet(width=ck.get("width", 24)).to(device)\n    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})\n    return m.eval()\n\n\nif __name__ == "__main__":\n    ap = argparse.ArgumentParser()\n    ap.add_argument("cmd")\n    ap.add_argument("--exclude", default=None)\n    ap.add_argument("--out", required=True)\n    ap.add_argument("--epochs", type=int, default=12)\n    ap.add_argument("--bs", type=int, default=64)\n    ap.add_argument("--seed", type=int, default=0)\n    a = ap.parse_args()\n    if a.cmd == "train":\n        train(a)\n'
try:
    import types as _t, lightgbm as _lgb, zarr as _zarr, pandas as _pd, torch as _torch, glob as _glob
    _M = _t.ModuleType("divclf_all")
    _M.__dict__["__name__"] = "divclf_all"
    for _s in (_SRC_DIVCLF, _SRC_DIVCLF2, _SRC_DIVCLF3):
        exec(_s, _M.__dict__)
    _CNN = _t.ModuleType("mitosis_cnn"); exec(_SRC_CNN.split('if __name__ == "__main__":')[0], _CNN.__dict__)
    _wdir = None
    for _cand in _glob.glob("/kaggle/input/**/cnn_all.pt", recursive=True):
        if "biohub-divclf3-weights" in _cand:
            _wdir = Path(_cand).parent; break
    if _wdir is None:
        raise FileNotFoundError("weights dataset biohub-divclf3-weights not attached")
    _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    _cnn = _CNN.load_model(_wdir / "cnn_all.pt", _dev)
    _boosters = [_lgb.Booster(model_file=str(_wdir / f"divclf3_s{s}.txt")) for s in range(3)]
    _boosters_r = [_lgb.Booster(model_file=str(_wdir / f"divclf3r_s{s}.txt")) for s in range(3) if (_wdir / f"divclf3r_s{s}.txt").exists()]
    _MIX = os.environ.get("BIOHUB_DIVCLF_MIX", "1") != "0" and len(_boosters_r) == 3
    print("DIVCLF3 models: base", len(_boosters), "readd", len(_boosters_r), "mix", _MIX, flush=True)

    _ET_W = float(os.environ.get("BIOHUB_DIVCLF_ET_W", "0"))
    _ET = np.load(_wdir / "divclf_et.npz") if (_ET_W > 0 and (_wdir / "divclf_et.npz").exists()) else None
    print("DIVCLF3 ExtraTrees weight", _ET_W, "loaded", _ET is not None, flush=True)

    def _et_predict(X):
        f, th, lf, rt, vl, of = _ET["feature"], _ET["threshold"], _ET["left"], _ET["right"], _ET["value"], _ET["offsets"]
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
        out = np.zeros(len(X)); rows = np.arange(len(X))
        for k in range(len(of) - 1):
            base = of[k]; node = np.zeros(len(X), np.int64)
            while True:
                g = base + node; leaf = lf[g] == -1
                if leaf.all():
                    break
                idx = rows[~leaf]; gi = g[~leaf]
                node[idx] = np.where(X[idx, f[gi]] <= th[gi], lf[gi], rt[gi])
            out += vl[base + node]
        return out / (len(of) - 1)

    def _score(X):
        s0 = np.mean([b.predict(X) for b in _boosters], axis=0)
        if _MIX:
            s0 = 0.5 * s0 + 0.5 * np.mean([b.predict(X) for b in _boosters_r], axis=0)
        if _ET is not None:
            s0 = (1.0 - _ET_W) * s0 + _ET_W * _et_predict(X)
        return s0
    PZ, PY, PX = 7, 32, 32
    _VARIANT = "v3"
    _FEATS5 = _M.FEATS3 + ["o_linked", "d_o_oldpar", "op_par", "op_dau", "blen_op"]
    _GATE_PAR, _GATE_DAU = 0.1, 0.1
    def _design(c):
        if _VARIANT == "v3":
            return _M.design3(c)
        X = c[_FEATS5].astype(float).copy(); X[_M.LOG3] = np.log(X[_M.LOG3].clip(lower=1e-3)); return X

    def _extract(v_frames, q, t, z, y, x, T):
        lo, hi = q["0.01"], q["0.999"]
        out = np.zeros((3, PZ, PY, PX), np.float32)
        for k, tt in enumerate((t - 1, t, t + 1)):
            tt = min(max(tt, 0), T - 1); v = v_frames(tt)
            z0, y0, x0 = z - PZ // 2, y - PY // 2, x - PX // 2
            zs = slice(max(z0, 0), min(z0 + PZ, v.shape[0])); ys = slice(max(y0, 0), min(y0 + PY, v.shape[1])); xs = slice(max(x0, 0), min(x0 + PX, v.shape[2]))
            out[k, zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = (v[zs, ys, xs].astype(np.float32) - lo) / (hi - lo)
        return out

    def _node_probs(nodes, ids, img, q):
        P = nodes.set_index("node_id")[["t", "z", "y", "x"]]
        ids = [i for i in sorted(set(ids)) if i in P.index]
        if not ids:
            return {}
        sub = P.loc[ids].sort_values("t")
        cache = {}
        def fr(t):
            if t not in cache:
                if len(cache) > 4: cache.pop(next(iter(cache)))
                cache[t] = img[t]
            return cache[t]
        X = np.zeros((len(sub), 3, PZ, PY, PX), np.float16)
        for k, (n, r) in enumerate(sub.iterrows()):
            X[k] = _extract(fr, q, int(r.t), int(r.z), int(r.y), int(r.x), img.shape[0]).astype(np.float16)
        pr = _CNN.predict(_cnn, X, _dev)
        return dict(zip(sub.index.tolist(), map(tuple, pr)))

    _sub_path = WORKING_DIR / "submission.csv"
    _sub = _pd.read_csv(_sub_path)
    _parts, _added, _pruned = [], 0, 0
    for _name, _grp in _sub.groupby("dataset", sort=False):
        _nodes = _grp[_grp.row_type == "node"][["node_id", "t", "z", "y", "x"]]
        _edges = _grp[_grp.row_type == "edge"][["source_id", "target_id"]].astype(int)
        _parts.append(_grp[_grp.row_type == "node"].drop(columns=["id"]))
        try:
            _zp = TEST_DIR / f"{_name}.zarr"
            _q = dict(_zarr.open_group(str(_zp), mode="r").attrs)["image_statistics"]["quantiles"]
            _img = _zarr.open_array(str(_zp / "0"), mode="r")
            _ft = _M.forks_table(_nodes, _edges)
            _prv = dict(zip(_edges.target_id, _edges.source_id))
            if _VARIANT == "v5":
                _probs = _node_probs(_nodes, _nodes.node_id.tolist(), _img, _q)
                _c = _M.candidates(_nodes, _edges, parent_um=17.0, include_linked=True)
                if len(_c):
                    _par = np.array([_probs.get(n, (0, 0, 0))[1] for n in _c.p]); _dau = np.array([_probs.get(n, (0, 0, 0))[2] for n in _c.o])
                    _c = _c[(_c.o_linked.values == 0) | ((_par >= _GATE_PAR) & (_dau >= _GATE_DAU))].reset_index(drop=True)
                    _op = [_prv.get(o, -1) for o in _c.o]
                    _c["op_par"] = [_probs[x][1] if x in _probs else 0.0 for x in _op]
                    _c["op_dau"] = [_probs[x][2] if x in _probs else 0.0 for x in _op]
                    _c["blen_op"] = [_M._blen(x, _prv) if x != -1 else 0 for x in _op]
                if len(_ft):
                    _ft["o_linked"] = 1; _ft["d_o_oldpar"] = _ft.d_po; _ft["op_par"] = _ft.p.map(lambda n: _probs.get(n, (0, 0, 0))[1]); _ft["op_dau"] = _ft.p.map(lambda n: _probs.get(n, (0, 0, 0))[2]); _ft["blen_op"] = _ft.len_p_back
            else:
                _c = _M.candidates(_nodes, _edges)
                _ids = set(_c.p) | set(_c.o) | set(_c.c) | set(_ft.p) | set(_ft.o) | set(_ft.c)
                _ids |= {_prv[p] for p in (set(_c.p) | set(_ft.p)) if p in _prv}
                _probs = _node_probs(_nodes, _ids, _img, _q)
            _new = _pd.DataFrame(columns=["source_id", "target_id"])
            if len(_c):
                _c = _M.add_image_features(_c, _nodes, _img, _q); _c = _M.add_temporal_features(_c, _nodes, _edges, _img, _q)
                _c = _M.add_cnn_features(_c, _edges, _probs)
                _X = _design(_c); _s = _score(_X.values)
                _new = _M.select(_c, _s, DIVCLF_THRESHOLD)
                if DIVCLF_MAX_ADD > 0 and len(_new) > DIVCLF_MAX_ADD:
                    _new = _new.head(DIVCLF_MAX_ADD)  # select() returns highest score first
            _e2 = _edges
            if DIVCLF_PRUNE_THRESHOLD > 0 and len(_ft):
                _ft = _M.add_image_features(_ft, _nodes, _img, _q); _ft = _M.add_temporal_features(_ft, _nodes, _edges, _img, _q)
                _ft = _M.add_cnn_features(_ft, _edges, _probs)
                _fs = _score(_design(_ft).values)
                _rm = set(zip(_ft.p[_fs < DIVCLF_PRUNE_THRESHOLD], _ft.o[_fs < DIVCLF_PRUNE_THRESHOLD]))
                _e2 = _edges[[(s, t) not in _rm for s, t in zip(_edges.source_id, _edges.target_id)]]
                _pruned += len(_rm)
            _added += len(_new)
            if _VARIANT == "v5" and len(_new):
                _e2 = _e2[~_e2.target_id.isin(set(_new.target_id))]  # stolen daughters lose their old parent edge
            _parts.append(_pd.concat([_e2, _new]).assign(dataset=_name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))
            print(f"DIVCLF3 {_name}: cands={len(_c)} forks={len(_ft)} +{len(_new)} -{len(_edges) - len(_e2)}", flush=True)
        except Exception as _e1:
            print("DIVCLF3 dataset failed, left unchanged:", _name, type(_e1).__name__, _e1, flush=True)
            _dc_tb.print_exc()
            _parts.append(_edges.assign(dataset=_name, row_type="edge", node_id=-1, t=-1, z=-1, y=-1, x=-1))
    _out = _pd.concat(_parts, ignore_index=True)[["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]]
    # final sanitation: drop edges whose endpoints are not nodes of the same dataset, and duplicate edges
    _keep = []
    for _name, _g in _out.groupby("dataset", sort=False):
        _nid = set(_g.loc[_g.row_type == "node", "node_id"].astype(int))
        _e = _g[_g.row_type == "edge"]
        _ok = _e.source_id.astype(int).isin(_nid) & _e.target_id.astype(int).isin(_nid) & ~_e.duplicated(subset=["source_id", "target_id"])
        _dropped = int((~_ok).sum())
        if _dropped:
            print(f"DIVCLF3 sanitize {_name}: dropped {_dropped} dangling/duplicate edges", flush=True)
        _keep.append(_g[_g.row_type == "node"]); _keep.append(_e[_ok])
    _out = _pd.concat(_keep, ignore_index=True)
    _out.insert(0, "id", np.arange(len(_out)))
    for _col in ["node_id", "t", "source_id", "target_id"]:
        _out[_col] = _out[_col].astype("int64")
    assert list(_out.columns) == list(_sub.columns)
    assert (_out.row_type == "node").sum() == (_sub.row_type == "node").sum()
    _out.to_csv(_sub_path, index=False)
    print(f"DIVCLF3 applied: +{_added} forks, -{_pruned} pruned; rows {len(_sub)} -> {len(_out)}", flush=True)
except Exception as _dc_exc:
    print("DIVCLF3 FAILED (submission left unchanged):", type(_dc_exc).__name__, _dc_exc, flush=True)
    _dc_tb.print_exc()

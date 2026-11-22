"""OOF CNN cell-state probs for ALL ILP candidate detections (edge_cache coords) of the 98 datasets.
Output divres/ilp_probs/{ds}.npz with p_cont,p_par,p_dau aligned to coords rows."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

sys.path.insert(0, "."); sys.path.insert(0, "../det"); sys.path.insert(0, "../eda")
from mitosis_cnn import load_model, predict  # noqa: E402
from mitosis_data import extract  # noqa: E402
from io_utils import DATA  # noqa: E402

CACHE = Path("exps/r_base98/kaggle/working/edge_cache")
OUT = Path("divres/ilp_probs"); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
folds = {k: set(Path(f"../det/folds/fold{k}.txt").read_text().split()) for k in range(5)}
ds2fold = {n: k for k, s in folds.items() for n in s}
models = {k: load_model(Path(f"../det/runs/cnn_f{k}.pt"), dev) for k in range(5)}


def coords_probs(coords, img, q, model, bs=2048):
    order = np.argsort(coords[:, 0], kind="stable")
    cache = {}
    def fr(t):
        if t not in cache:
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
            cache[t] = img[t]
        return cache[t]
    class Im:
        shape = img.shape
        def __getitem__(self, t):
            return fr(t)
    im = Im()
    out = np.zeros((len(coords), 3), np.float32)
    X = np.zeros((bs, 3, 7, 32, 32), np.float32)
    k = 0; idx = []
    for i in order:
        t, z, y, x = coords[i]
        X[k] = extract(im, q, int(t), int(z), int(y), int(x)); idx.append(i); k += 1
        if k == bs:
            out[idx] = predict(model, X, dev); k = 0; idx = []
    if k:
        out[idx] = predict(model, X[:k], dev)
    return out


if __name__ == "__main__":
    names = sorted(p.stem for p in CACHE.glob("*.npz"))
    for name in names:
        f = OUT / f"{name}.npz"
        if f.exists():
            continue
        coords = np.load(CACHE / f"{name}.npz")["coords"].astype(int)
        zp = DATA / "train" / f"{name}.zarr"
        q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
        img = zarr.open_array(str(zp / "0"), mode="r")
        pr = coords_probs(coords, img, q, models[ds2fold[name]])
        np.savez_compressed(f, p_cont=pr[:, 0], p_par=pr[:, 1], p_dau=pr[:, 2])
        print(name, len(coords), "p_par>0.3:", int((pr[:, 1] > 0.3).sum()), flush=True)

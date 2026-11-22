"""Build a patch dataset for a mitosis (cell-state) classifier from GT tracks.

Classes: 0 = continue, 1 = parent (node with 2 GT children; also the frame before, label 1 with 'soft' flag),
         2 = daughter (node whose GT parent divided; also the following frame).
Patch: 3 frames (t-1, t, t+1) x (PZ, PY, PX) voxels centred on the node, normalised with the zarr quantiles.
Output: det/mitosis_patches.npz  (X: float16 [N,3,PZ,PY,PX], y, ds, t, id)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from io_utils import DATA, load_geff  # noqa: E402

PZ, PY, PX = 7, 32, 32
NEG_PER_DS = 120
RNG = np.random.default_rng(0)


def extract(img, q, t, z, y, x):
    lo, hi = q["0.01"], q["0.999"]
    T = img.shape[0]
    out = np.zeros((3, PZ, PY, PX), np.float32)
    for k, tt in enumerate((t - 1, t, t + 1)):
        tt = min(max(tt, 0), T - 1)
        v = img[tt]
        z0, y0, x0 = z - PZ // 2, y - PY // 2, x - PX // 2
        zs, ys, xs = slice(max(z0, 0), min(z0 + PZ, v.shape[0])), slice(max(y0, 0), min(y0 + PY, v.shape[1])), slice(max(x0, 0), min(x0 + PX, v.shape[2]))
        sub = (v[zs, ys, xs].astype(np.float32) - lo) / (hi - lo)
        out[k, zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = sub
    return out


def run(geff):
    name = geff.name[:-5]
    N, E, _ = load_geff(geff)
    N = N.set_index("id")
    kids = E.groupby("src").dst.apply(list).to_dict()
    par = dict(zip(E.dst, E.src))
    parents = [n for n, k in kids.items() if len(k) >= 2]
    daughters = [k for p in parents for k in kids[p]]
    rows = []
    for p in parents:
        rows.append((p, 1, 0))
        if p in par:
            rows.append((par[p], 1, 1))  # frame before division (soft)
    for d in daughters:
        rows.append((d, 2, 0))
        if len(kids.get(d, [])) == 1:
            rows.append((kids[d][0], 2, 1))
    special = {r[0] for r in rows}
    cont = [n for n in N.index if n not in special and n in par and len(kids.get(n, [])) == 1]
    for n in RNG.choice(cont, size=min(NEG_PER_DS, len(cont)), replace=False):
        rows.append((n, 0, 0))
    zp = DATA / "train" / f"{name}.zarr"
    q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
    img = zarr.open_array(str(zp / "0"), mode="r")
    X, y, soft, ts, ids = [], [], [], [], []
    for n, lab, sf in rows:
        r = N.loc[n]
        X.append(extract(img, q, int(r.t), int(r.z), int(r.y), int(r.x)).astype(np.float16))
        y.append(lab); soft.append(sf); ts.append(int(r.t)); ids.append(int(n))
    return name, np.stack(X), np.array(y), np.array(soft), np.array(ts), np.array(ids)


if __name__ == "__main__":
    geffs = sorted((DATA / "train").glob("*.geff"))
    Xs, ys, ss, ds, ts, ids = [], [], [], [], [], []
    with ProcessPoolExecutor(6) as ex:
        for name, X, y, s, t, i in ex.map(run, geffs):
            Xs.append(X); ys.append(y); ss.append(s); ds += [name] * len(y); ts.append(t); ids.append(i)
            print(name, len(y), np.bincount(y, minlength=3), flush=True)
    np.savez_compressed(Path(__file__).parent / "mitosis_patches.npz", X=np.concatenate(Xs), y=np.concatenate(ys),
                        soft=np.concatenate(ss), ds=np.array(ds), t=np.concatenate(ts), id=np.concatenate(ids))
    print("total", sum(len(y) for y in ys))

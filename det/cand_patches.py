"""Extract 3-frame patches for all unique candidate nodes (p, o, c, prev(p)) of the divclf candidate table.
usage: python cand_patches.py cands.parquet sub.csv out_dir   -> out_dir/{ds}.npz (ids, X float16)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
from io_utils import DATA  # noqa: E402
from mitosis_data import extract  # noqa: E402


def run(args):
    name, ids, out = args
    if (out / f"{name}.npz").exists():
        return name, 0
    sub = pd.read_csv(SUB)
    g = sub[(sub.dataset == name) & (sub.row_type == "node")].set_index("node_id")[["t", "z", "y", "x"]]
    zp = DATA / "train" / f"{name}.zarr"
    q = dict(zarr.open_group(str(zp), mode="r").attrs)["image_statistics"]["quantiles"]
    img = zarr.open_array(str(zp / "0"), mode="r")
    ids = np.array(sorted(set(ids)))
    X = np.zeros((len(ids), 3, 7, 32, 32), np.float16)
    # cache frames: iterate by t
    P = g.loc[ids]
    cache = {}
    def frame(t):
        t = min(max(t, 0), img.shape[0] - 1)
        if t not in cache:
            if len(cache) > 4:
                cache.pop(next(iter(cache)))
            cache[t] = img[t]
        return cache[t]
    class Img:
        shape = img.shape
        def __getitem__(self, t):
            return frame(t)
    im = Img()
    order = np.argsort(P.t.values)
    for k in order:
        r = P.iloc[k]
        X[k] = extract(im, q, int(r.t), int(r.z), int(r.y), int(r.x)).astype(np.float16)
    np.savez_compressed(out / f"{name}.npz", ids=ids, X=X)
    return name, len(ids)


if __name__ == "__main__":
    cands, SUB, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    C = pd.read_parquet(cands)
    sub = pd.read_csv(SUB)
    E = sub[sub.row_type == "edge"]
    jobs = []
    for name, c in C.groupby("ds"):
        e = E[E.dataset == name]
        prv = dict(zip(e.target_id.astype(int), e.source_id.astype(int)))
        ids = set(c.p) | set(c.o) | set(c.c) | {prv[p] for p in set(c.p) if p in prv}
        jobs.append((name, list(ids), out))
    with ProcessPoolExecutor(4) as ex:
        for name, n in ex.map(run, jobs):
            print(name, n, flush=True)

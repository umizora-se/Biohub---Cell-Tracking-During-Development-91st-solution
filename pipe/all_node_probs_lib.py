import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../det")
from mitosis_cnn import predict
from mitosis_data import extract


def node_probs(nodes, img, q, model, dev, bs=2048):
    """nodes: DataFrame indexed by node_id with t,z,y,x. Returns DataFrame[node_id, p_cont, p_par, p_dau]."""
    P = nodes.sort_values("t")
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
    out = []
    X = np.zeros((bs, 3, 7, 32, 32), np.float32)
    k = 0
    for n, r in P.iterrows():
        X[k] = extract(im, q, int(r.t), int(r.z), int(r.y), int(r.x)); k += 1
        if k == bs:
            out.append(predict(model, X, dev)); k = 0
    if k:
        out.append(predict(model, X[:k], dev))
    pr = np.concatenate(out)
    return pd.DataFrame({"node_id": P.index.to_numpy(), "p_cont": pr[:, 0], "p_par": pr[:, 1], "p_dau": pr[:, 2]})



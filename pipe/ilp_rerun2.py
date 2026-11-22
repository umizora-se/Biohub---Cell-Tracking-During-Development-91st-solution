"""Cell-state-aware ILP re-solve (Hirsch et al. 2022 idea) from cached inputs.

edge_prob' = clip(edge_prob * (1 - beta * p_dau(tgt) * (1 - p_par(src))) + alpha * p_par(src) * p_dau(tgt), 0, 1)
division cost per node = div_w * (1 - p_par)^gamma      (default div_w 1.2 like the notebook)

usage: python ilp_rerun2.py SRC_EXP DST_EXP --alpha 0.5 --beta 0.5 --gamma 1 [--divw 1.2] [--jobs 4] [--only ...]
Reads SRC_EXP/kaggle/working/edge_cache/*.npz and divres/ilp_probs/{ds}.npz; writes predictions to DST_EXP.
"""
import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PIPE = Path(__file__).resolve().parent
PRED_SUB = "unet_transformer/split_0"  # under tracking_repo/predictions/<user>/ (user dir taken from the source experiment)


def solve(args):
    npz, probs_f, out_geff, w, repo_src = args
    import contextlib
    import io

    import numpy as np
    import polars as pl
    import tracksdata as td
    from tracksdata.attrs import NodeAttr
    sys.path.insert(0, repo_src)
    from biohub_tracking.io import save_graph

    z = np.load(npz)
    coords, adm = z["coords"], z["admitted"]
    pr = np.load(probs_f)
    p_par, p_dau = pr["p_par"].astype(float), pr["p_dau"].astype(float)
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x", "p_par", "p_dau", "div_cost"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    div_cost = w["divw"] * (1.0 - p_par) ** w["gamma"] - w["rho"] * p_par
    ids = g.bulk_add_nodes([{"t": int(t), "z": float(a), "y": float(b), "x": float(c), "p_par": float(pp), "p_dau": float(pd_), "div_cost": float(dc)}
                            for (t, a, b, c), pp, pd_, dc in zip(coords, p_par, p_dau, div_cost)])
    n_mod = 0
    n_inj = 0
    if w.get("inject", 0) > 0 and len(adm):
        # inject candidate edges parent-like(t) -> daughter-like(t+1) within inject_um that are not already candidates
        from scipy.spatial import cKDTree
        have = {(int(s), int(t)) for s, t, p, d in adm}
        SC = np.array([1.625, 0.40625, 0.40625])
        T = coords[:, 0].astype(int)
        extra = []
        for t in np.unique(T):
            src = np.where((T == t) & (p_par >= w["inj_par"]))[0]
            dst = np.where((T == t + 1) & (p_dau >= w["inj_dau"]))[0]
            if len(src) == 0 or len(dst) == 0:
                continue
            tree = cKDTree(coords[dst, 1:].astype(float) * SC)
            for i in src:
                for j in tree.query_ball_point(coords[i, 1:].astype(float) * SC, w["inject_um"]):
                    jj = dst[j]
                    if (i, jj) in have:
                        continue
                    dist = float(np.linalg.norm((coords[i, 1:] - coords[jj, 1:]).astype(float) * SC))
                    extra.append((i, jj, w["inject"] * p_par[i] * p_dau[jj], dist))
        if extra:
            adm = np.concatenate([adm, np.array(extra, dtype=float)])
            n_inj = len(extra)
    if len(adm):
        g.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        g.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        rows = []
        for s, t, p, d in adm:
            s, t = int(s), int(t)
            p2 = p * (1.0 - w["beta"] * p_dau[t] * (1.0 - p_par[s])) + w["alpha"] * p_par[s] * p_dau[t]
            p2 = min(max(p2, 0.0), 1.0)
            n_mod += abs(p2 - p) > 0.05
            rows.append({"source_id": ids[s], "target_id": ids[t], "edge_prob": float(p2), "edge_dist": float(d)})
        g.bulk_add_edges(rows)
        solver = td.solvers.ILPSolver(edge_weight=-1.0 * td.EdgeAttr("edge_prob"), appearance_weight=w["app"],
                                      disappearance_weight=w["disapp"], division_weight=NodeAttr("div_cost"))
        with contextlib.redirect_stdout(io.StringIO()):
            g = solver.solve(g)
    save_graph(g, out_geff)
    return Path(npz).stem, g.num_nodes(), g.num_edges(), int(n_mod) + 1000000 * n_inj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--divw", type=float, default=1.2)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--inject", type=float, default=0.0, help="prob scale for injected parent->daughter candidate edges (0=off)")
    ap.add_argument("--inject-um", type=float, default=14.0)
    ap.add_argument("--inj-par", type=float, default=0.3)
    ap.add_argument("--inj-dau", type=float, default=0.3)
    ap.add_argument("--app", type=float, default=0.0)
    ap.add_argument("--disapp", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    sw = PIPE / "exps" / a.src / "kaggle/working"
    dw = PIPE / "exps" / a.dst / "kaggle/working"
    src_pred = next((sw / "tracking_repo/predictions").glob(f"*/{PRED_SUB}"))
    pred_rel = src_pred.relative_to(sw)
    (dw / pred_rel).mkdir(parents=True, exist_ok=True)
    shutil.copytree(sw / "edge_cache", dw / "edge_cache", dirs_exist_ok=True)
    npzs = sorted((sw / "edge_cache").glob("*.npz"))
    if a.only:
        npzs = [p for p in npzs if p.stem in a.only]
    w = dict(alpha=a.alpha, beta=a.beta, gamma=a.gamma, divw=a.divw, rho=a.rho, app=a.app, disapp=a.disapp, inject=a.inject, inject_um=a.inject_um, inj_par=a.inj_par, inj_dau=a.inj_dau)
    jobs = [(str(p), str(PIPE / "divres/ilp_probs" / f"{p.stem}.npz"), str(dw / pred_rel / f"{p.stem}.geff"), w, str(sw / "tracking_repo/src")) for p in npzs]
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, nn, ne, nm in ex.map(solve, jobs):
            print(f"{name}: {nn} nodes {ne} edges (modified edges {nm % 1000000}, injected {nm // 1000000})", flush=True)


if __name__ == "__main__":
    main()

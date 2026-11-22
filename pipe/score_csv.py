"""Score a submission CSV against train GT with the official metric + error breakdown.

usage: python score_csv.py submission.csv [out_rows.csv]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref_royerlab" / "scripts"))
from csv_to_geffs import build_graph_from_rows  # noqa: E402
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # noqa: E402

warnings.filterwarnings("ignore")
TRAIN = Path(__file__).resolve().parents[1] / "data" / "train"
SCALE = np.array([1.625, 0.40625, 0.40625])


def load_gt(name):
    g = td.graph.IndexedRXGraph.from_geff(TRAIN / f"{name}.geff")
    g = g[0] if isinstance(g, tuple) else g
    n_total = GeffMetadata.read(TRAIN / f"{name}.geff").extra["estimated_number_of_nodes"]
    return g, float(n_total)


def breakdown(nodes, edges, gnodes, gedges):
    """FN edges: missing endpoint detection vs linking error (own matcher, same 7um rule)."""
    p2g, g2p = {}, {}
    for t, gs in gnodes.group_by("t"):
        ps = nodes.filter(pl.col("t") == t[0])
        if ps.height == 0:
            continue
        a = gs.select("z", "y", "x").to_numpy() * SCALE
        b = ps.select("z", "y", "x").to_numpy() * SCALE
        d = np.linalg.norm(a[:, None] - b[None], axis=-1)
        d[d > 7] = 1e6
        i, j = linear_sum_assignment(d)
        for ii, jj in zip(i, j):
            if d[ii, jj] < 1e6:
                g2p[gs["id"][int(ii)]] = ps["node_id"][int(jj)]
    pe = set(zip(edges["source_id"].to_list(), edges["target_id"].to_list()))
    miss_node = wrong_link = ok = 0
    for s, t in zip(gedges["source_id"].to_list(), gedges["target_id"].to_list()):
        if s not in g2p or t not in g2p:
            miss_node += 1
        elif (g2p[s], g2p[t]) in pe:
            ok += 1
        else:
            wrong_link += 1
    return dict(gt_edges=len(gedges), ok=ok, fn_missing_node=miss_node, fn_wrong_link=wrong_link,
                gt_node_recall=len(g2p) / max(len(gnodes), 1))


def main(csv, out=None):
    df = pl.read_csv(csv)
    rows = []
    for (name,), grp in sorted(df.group_by("dataset"), key=lambda x: x[0]):
        nrows = grp.filter(pl.col("row_type") == "node")
        erows = grp.filter(pl.col("row_type") == "edge")
        pred = build_graph_from_rows(nrows, erows)
        gt, n_total = load_gt(name)
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        r = per_sample_metrics(er, n_total, node_recall(pred, gt))
        gn = gt.node_attrs().select(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).alias("id"), "t", "z", "y", "x")
        ge = gt.edge_attrs().select("source_id", "target_id")
        r.update(breakdown(nrows, erows, gn, ge))
        r.update(ds=name, n_total=n_total)
        rows.append(r)
        print(f"{name}: J={r['edge_jaccard']:.4f} adj={r['adj_edge_jaccard']:.4f} ratio={r['total_node_ratio']:+.3f} "
              f"tp/fp/fn={r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']} (miss_node={r['fn_missing_node']} wrong_link={r['fn_wrong_link']}) "
              f"div={r['division_tp']}/{r['division_fp']}/{r['division_fn']}", flush=True)
    s = summarise(rows)
    res = pl.DataFrame(rows)
    for g in ("44b6", "6bba"):
        sub = [r for r in rows if r["ds"].startswith(g)]
        if sub:
            ss = summarise(sub)
            print(f"[{g}] score={ss['score']:.4f} adj={ss['adj_edge_jaccard']:.4f} edgeJ={ss['edge_jaccard']:.4f} divJ={ss['division_jaccard']:.4f}")
    tot = res.select(pl.col("fn_missing_node").sum(), pl.col("fn_wrong_link").sum(), pl.col("edge_fp").sum(), pl.col("edge_tp").sum())
    print(tot)
    print(f"SCORE={s['score']:.4f} adj_edge={s['adj_edge_jaccard']:.4f} edgeJ={s['edge_jaccard']:.4f} "
          f"divJ={s['division_jaccard']:.4f} (div tp/fp/fn={s['division_tp']}/{s['division_fp']}/{s['division_fn']}) "
          f"mean_ratio={res['total_node_ratio'].mean():+.3f}")
    if out:
        res.write_csv(out)


if __name__ == "__main__":
    main(*sys.argv[1:])

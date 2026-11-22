"""Figures with real images (Figure 1: data overview, Figure 3: division example) for the write-up.
Needs the competition data (repro/01_download.sh) and the base-pipeline output (repro/02_base.sh).
Run from the project root:  .venv/bin/python writeup/make_figs_cells.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eda"))
from io_utils import find_root, load_geff, load_image  # noqa: E402

OUT = ROOT / "writeup" / "figs"
DATA = find_root()
SUB = ROOT / "pipe/exps/r_base98/kaggle/working/submission.csv"  # output of repro/02_base.sh
BLUE, ORANGE, AQUA, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#fcfcfb"
SC = np.array([1.625, 0.40625, 0.40625])

plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 11,
                     "axes.titlesize": 11.5, "text.color": "#0b0b0b"})


def norm(v, q):
    lo, hi = q["0.01"], q["0.999"]
    return np.clip((v.astype(np.float32) - lo) / (hi - lo), 0, 1) ** 0.7


def quant(ds):
    import zarr
    a = dict(zarr.open_group(str(DATA / "train" / f"{ds}.zarr"), mode="r").attrs)
    return {str(k): float(v) for k, v in a["image_statistics"]["quantiles"].items()}


# ------------------------------------------------------------- fig 0: overview
def fig_overview():
    picks = [("44b6_0b24845f", "line 44b6: dense, low contrast"), ("6bba_05b6850b", "line 6bba: sparse, high contrast")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, (ds, title) in zip(axes, picks):
        img = load_image(DATA / "train" / f"{ds}.zarr")
        q = quant(ds)
        t = 50
        mip = norm(np.asarray(img[t]).max(0), q)
        ax.imshow(mip, cmap="gray", extent=[0, mip.shape[1] * SC[2], mip.shape[0] * SC[1], 0])
        N, _, _ = load_geff(DATA / "train" / f"{ds}.geff")
        n = N[N.t == t]
        ax.scatter(n.x * SC[2], n.y * SC[1], s=40, facecolor="none", edgecolor=ORANGE, lw=1.4,
                   label=f"annotated nuclei at t={t} ({len(n)})")
        ax.set_title(f"{title}\n{ds}, frame {t}, max projection over z", fontsize=11)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.legend(loc="lower right", fontsize=9, facecolor="white", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "fig0_data_overview.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fig0")


# ----------------------------------------------------- fig 7: a rescued division
def fig_division(ds="6bba_61ecbe65", t0=34, p=2539, o=2597, c=2598, half=48, frames=(-1, 0, 1, 2, 3)):
    S = pd.read_csv(SUB, usecols=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
    S = S[S.dataset == ds]
    nodes = S[S.row_type == "node"].set_index("node_id")
    edges = S[S.row_type == "edge"][["source_id", "target_id"]].astype(int)
    nxt = edges.groupby("source_id").target_id.apply(list).to_dict()
    prv = dict(zip(edges.target_id, edges.source_id))

    def chain(start, fwd, n):
        out, cur = [start], start
        for _ in range(n):
            cur = (nxt.get(cur, [None])[0] if fwd else prv.get(cur))
            if cur is None:
                break
            out.append(cur)
        return out

    par_track = chain(p, False, 2)[::-1]           # ... -> p
    c_track, o_track = chain(c, True, 3), chain(o, True, 3)
    role = {**{n: "parent" for n in par_track}, **{n: "c" for n in c_track}, **{n: "o" for n in o_track}}
    col = {"parent": BLUE, "c": AQUA, "o": ORANGE}

    img = load_image(DATA / "train" / f"{ds}.zarr")
    q = quant(ds)
    zc, yc, xc = (int(round(nodes.at[p, k])) for k in ("z", "y", "x"))
    G, GE, _ = load_geff(DATA / "train" / f"{ds}.geff")

    fig, axes = plt.subplots(1, len(frames), figsize=(3.0 * len(frames), 3.6))
    for ax, dt in zip(axes, frames):
        t = t0 + dt
        vol = np.asarray(img[t, max(zc - 4, 0):zc + 5, yc - half:yc + half, xc - half:xc + half])
        ax.imshow(norm(vol.max(0), q), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        g = G[(G.t == t) & (abs(G.y - yc) < half) & (abs(G.x - xc) < half)]
        ax.scatter(g.x - xc + half, g.y - yc + half, marker="+", s=70, color="white", lw=1.4, zorder=5)
        for n, r in role.items():
            if n in nodes.index and int(nodes.at[n, "t"]) == t:
                ax.scatter(nodes.at[n, "x"] - xc + half, nodes.at[n, "y"] - yc + half, s=260, facecolor="none",
                           edgecolor=col[r], lw=2.2, zorder=4)
        ax.set_title(f"t = {t}" + ("  (parent p)" if dt == 0 else "  (daughters)" if dt == 1 else ""))
        if dt == 0:
            ax.plot([half - 8, half - 8 + 10 / SC[2]], [2 * half - 6] * 2, color="white", lw=3)
            ax.text(half - 8 + 5 / SC[2], 2 * half - 10, "10 µm", color="white", ha="center", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, mew=2, ms=11, label="parent track"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=AQUA, mew=2, ms=11, label="daughter c (linked by base)"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ORANGE, mew=2, ms=11,
                          label="daughter o (orphan start → rescued, score 0.99)"),
               plt.Line2D([], [], marker="+", ls="", color="#9a9a9a", ms=10, mew=1.5, label="ground-truth nucleus")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"A division the base pipeline missed and the division model recovered  ({ds}, "
                 f"39 µm crops, max projection over 9 z-slices)", fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_division_example.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fig7", {"parent": par_track, "c": c_track, "o": o_track})


if __name__ == "__main__":
    fig_overview(); fig_division()

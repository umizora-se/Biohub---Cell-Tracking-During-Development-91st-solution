"""Figures for the Kaggle solution write-up. Run from the project root:  python3 writeup/make_figs.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "writeup" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (light)
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.9); ax.axis("off")

    def box(x, y, w, h, title, body, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec=color, lw=1.8))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.72, body, ha="center", va="top", fontsize=9.3, color=INK2, linespacing=1.45)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK2))

    pub, mine = "#f1f0ec", "#e8f1fb"
    y, h, w = 1.0, 2.25, 2.35
    box(0.1, y, w, h, "Detection", "3D U-Net (public weights)\nflip TTA\ndual-seed ensemble\nthreshold 0.965", MUTED, pub)
    box(2.75, y, w, h, "Edge scoring", "temporal U-Net\nedge features + TTA\nharmonic fwd/bwd\nfusion", MUTED, pub)
    box(5.4, y, w, h, "ILP linking", "tracksdata / ilpy\ndivision cost 1.2\nappear 0 / disappear 2", MUTED, pub)
    box(8.05, y, w, h, "Post-processing", "gap closing, motion\nrelink (optical flow,\nFLOW_ITER 1 -> 3)\nmin track len 6", BLUE, mine)
    box(10.7, y, 2.2, h, "Division model", "3D-CNN cell state\n+ LightGBM\n+ ExtraTrees\nadd / prune forks", BLUE, mine)
    for x in (2.45, 5.1, 7.75, 10.4):
        arrow(x, y + h / 2, x + 0.3, y + h / 2)

    ax.add_patch(FancyBboxPatch((0.1, 0.25), 0.35, 0.3, boxstyle="round,pad=0.01", fc=pub, ec=MUTED, lw=1.4))
    ax.text(0.55, 0.4, "public notebook (0.953 LB), used as is", va="center", fontsize=9.5, color=INK2)
    ax.add_patch(FancyBboxPatch((4.6, 0.25), 0.35, 0.3, boxstyle="round,pad=0.01", fc=mine, ec=BLUE, lw=1.4))
    ax.text(5.05, 0.4, "my changes (+0.017 public, +0.010 private vs. base)", va="center", fontsize=9.5, color=INK2)
    ax.text(6.5, 3.85, "Final pipeline: biohub-exp-v3-iter3-et07", ha="center", va="top", fontsize=13, weight="bold")
    save(fig, "fig1_pipeline.png")


# ------------------------------------------------------- 2. division schematic
def fig_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.set_aspect("equal")

    def node(ax, x, y, label, color, txt=None):
        ax.add_patch(Circle((x, y), 0.32, fc=color, ec=SURF, lw=2, zorder=3))
        if txt:
            ax.text(x, y - 0.65, txt, ha="center", va="top", fontsize=9.5, color=INK2)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color="white", weight="bold", zorder=4)

    def link(ax, a, b, color=INK2, style="-", lw=2):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=style,
                                                        shrinkA=11, shrinkB=11), zorder=2)

    # (a) rescue
    ax = axes[0]
    ax.set_title("(a) Rescue: add a missing fork", loc="left")
    pts = [(1, 3), (2.6, 3), (4.2, 3)]
    for a, b in zip(pts, pts[1:]):
        link(ax, a, b)
    p = pts[-1]
    c1, c2 = (5.9, 4.0), (7.6, 4.0)
    o1, o2 = (5.9, 1.9), (7.6, 1.9)
    link(ax, p, c1); link(ax, c1, c2)
    link(ax, o1, o2)
    link(ax, p, o1, color=ORANGE, style="--", lw=2.4)
    for q in pts[:-1]:
        node(ax, *q, "", MUTED)
    node(ax, *p, "p", BLUE)
    ax.text(p[0] - 0.1, p[1] - 0.75, "parent\n(1 child)", ha="right", fontsize=9.5, color=INK2)
    node(ax, *c1, "c", AQUA)
    ax.text(c1[0], c1[1] + 0.5, "existing child", ha="center", fontsize=9.5, color=INK2)
    node(ax, *c2, "", AQUA)
    node(ax, *o1, "o", ORANGE, "orphan track start")
    node(ax, *o2, "", ORANGE)
    ax.text(3.9, 0.9, "classifier p ≥ 0.3\n→ add edge p→o", color=ORANGE, fontsize=9.5, ha="right")

    # (b) prune
    ax = axes[1]
    ax.set_title("(b) Prune: remove an unlikely fork", loc="left")
    pts = [(1, 3), (2.6, 3), (4.2, 3)]
    for a, b in zip(pts, pts[1:]):
        link(ax, a, b)
    p = pts[-1]
    k1, k2 = (5.9, 4.0), (5.9, 1.9)
    link(ax, p, k1)
    link(ax, p, k2, color=RED, style="--", lw=2.4)
    link(ax, k1, (7.6, 4.0)); link(ax, k2, (7.6, 1.9))
    for q in pts[:-1]:
        node(ax, *q, "", MUTED)
    node(ax, *p, "p", BLUE)
    ax.text(p[0] - 0.1, p[1] - 0.75, "parent with\n2 children", ha="right", fontsize=9.5, color=INK2)
    node(ax, *k1, "", AQUA); node(ax, 7.6, 4.0, "", AQUA)
    node(ax, *k2, "", MUTED); node(ax, 7.6, 1.9, "", MUTED)
    ax.text(3.9, 0.9, "classifier p < 0.05\n→ drop the fork edge", color=RED, fontsize=9.5, ha="right")
    ax.text(9.6, 0.35, "time →", ha="right", color=MUTED, fontsize=9.5)
    save(fig, "fig2_division_ops.png")


# --------------------------------------------------------- 3. OOF classifier curves
def fig_oof():
    C = pd.read_parquet(ROOT / "pipe/divres/oof3/cands_oof3.parquet")
    L = C[C.label >= 0]
    y = L.label.to_numpy()
    series = [("v2: geometry + intensity + temporal (LightGBM)", "oof_v2", ORANGE),
              ("v3: v2 + 3D-CNN cell-state probabilities", "oof_v3", BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for name, col, color in series:
        s = L[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name.split(':')[0]}  AUC {roc_auc_score(y, s):.3f}")
        axes[1].plot(rc, pr, color=color, lw=2, label=f"{name.split(':')[0]}  AP {average_precision_score(y, s):.3f}")
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    fig.suptitle(f"Division-candidate classifier, grouped out-of-fold on 98 movies "
                 f"({int(y.sum())} true divisions / {len(y):,} labelled candidates)",
                 x=0.06, ha="left", fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_oof_curves.png")


# ------------------------------------------------------------- 4. CNN probabilities
def fig_cnn():
    C = pd.read_parquet(ROOT / "pipe/divres/oof3/cands_oof3.parquet")
    L = C[C.label >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=False)
    bins = np.linspace(0, 1, 21)
    for ax, col, title in [(axes[0], "p_par", "CNN P(parent) at p"), (axes[1], "o_dau", "CNN P(daughter) at o")]:
        for lab, name, color in [(0, "not a division", MUTED), (1, "true division", BLUE)]:
            v = L.loc[L.label == lab, col]
            w = np.ones(len(v)) / len(v)
            ax.hist(v, bins=bins, weights=w, color=color, alpha=0.85 if lab else 0.6, label=f"{name} (n={len(v):,})",
                    edgecolor=SURF, linewidth=1.5)
        ax.set(title=title, xlabel="probability", ylabel="share of candidates")
        ax.legend(loc="upper center")
    fig.tight_layout()
    save(fig, "fig4_cnn_probs.png")


# ---------------------------------------------------------- 5. public vs private
def fig_lb():
    S = pd.read_csv(ROOT / "writeup/submissions.csv")
    S = S.sort_values("date").reset_index(drop=True)
    final = {56658783: "final: et07 (LGB/ET 30:70)", 56647720: "2nd pick: LGB/ET 50:50"}
    best = S.privateScore.idxmax()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    sel = S.ref.isin(final)
    ax.scatter(S.publicScore[~sel], S.privateScore[~sel], s=70, color=MUTED, edgecolor=SURF, lw=2, zorder=3,
               label="other submissions")
    ax.scatter(S.publicScore[sel], S.privateScore[sel], s=110, color=BLUE, edgecolor=SURF, lw=2, zorder=4,
               label="selected for final")
    ax.scatter([S.publicScore[best]], [S.privateScore[best]], s=110, color=ORANGE, edgecolor=SURF, lw=2, zorder=4,
               label="best private (not selected)")
    notes = {56658783: ("final: et07\n0.970 / 0.929", (-110, 18)),
             56647720: ("2nd pick: LGB/ET 50:50\n0.970 / 0.928", (-40, -55)),
             56630232: ("cell-state ILP + readd forks\n+ classifier mix\n0.966 / 0.935", (-195, -8)),
             56661666: ("divclf retrained on\niter3 graph\n0.969 / 0.934", (10, -12)),
             56576855: ("base notebook + min len 4\n0.953 / 0.918", (10, -4)),
             56583002: ("first division model (v1)\n0.949 / 0.917", (10, -24))}
    for ref, (txt, off) in notes.items():
        r = S[S.ref == ref].iloc[0]
        ax.annotate(txt, (r.publicScore, r.privateScore), xytext=off, textcoords="offset points",
                    fontsize=8.8, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
    ax.set(xlabel="Public LB", ylabel="Private LB",
           title="All 18 submissions: public rank did not predict private rank")
    ax.legend(loc="upper left")
    save(fig, "fig5_public_vs_private.png")

    # timeline (two panels, same x, separate y – no dual axis)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    x = np.arange(1, len(S) + 1)
    for ax, col, title in [(axes[0], "publicScore", "Public LB"), (axes[1], "privateScore", "Private LB")]:
        ax.plot(x, S[col], color=BLUE if col == "publicScore" else AQUA, lw=2, marker="o", ms=6,
                markeredgecolor=SURF, markeredgewidth=1.5)
        for ref in final:
            i = S.index[S.ref == ref][0]
            ax.scatter(x[i], S[col][i], s=120, facecolor="none", edgecolor=INK, lw=1.6, zorder=5)
        ax.set_ylabel(title)
    axes[1].set_xlabel("submission # (chronological, 2026-09-26 → 09-29)")
    axes[0].set_title("Score per submission (circled = final selection)", loc="left")
    axes[1].set_xticks(x)
    fig.tight_layout()
    save(fig, "fig6_timeline.png")


if __name__ == "__main__":
    fig_pipeline(); fig_schematic(); fig_oof(); fig_cnn(); fig_lb()

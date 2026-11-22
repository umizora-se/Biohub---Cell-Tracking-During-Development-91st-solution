"""Build the three submitted Kaggle notebooks from the public base notebook + the cells of this solution.

usage: python sub/build_notebooks.py [--out DIR] [--owner KAGGLE_USER] [--weights-dataset OWNER/SLUG] [--check]

  --check  compare the built notebooks with notebooks/*/notebook.ipynb (the versions that were submitted)

The division-model cells find their weights with a glob over /kaggle/input/**/cnn_all.pt, so the weights
dataset can have any name; --weights-dataset only sets kernel-metadata.json.
Same construction as sub/make_kernel.py (overrides appended to cell 0, config-drift guard relaxed in cell 1,
extra cells inserted after the given cell index, string replacements applied to cell 5).
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "nb/biohub-0-953-lb-original"
CELLS = ROOT / "sub/cells_final"

COMMON = {"BIOHUB_MOTION_RELINK_FLOW_ITER": "3", "BIOHUB_REPAIR_DEADLINE_S": "36000", "BIOHUB_DIVCLF_MIX": "0"}
NOTEBOOKS = {
    # final submission: LightGBM/ExtraTrees 30:70, add >= 0.3, prune < 0.05
    "final_et07": dict(
        slug="biohub-exp-v3-iter3-et07",
        overrides={**COMMON, "BIOHUB_DIVCLF_ET_W": "0.7", "BIOHUB_DIVCLF_THRESHOLD": "0.3", "BIOHUB_DIVCLF_PRUNE_THRESHOLD": "0.05"},
        patches=[(11, "divclf3_et.py")], replaces={}),
    # second final pick: LightGBM/ExtraTrees 50:50
    "second_et05": dict(
        slug="biohub-exp-v3-iter3-et",
        overrides={**COMMON, "BIOHUB_DIVCLF_ET_W": "0.5", "BIOHUB_DIVCLF_THRESHOLD": "0.3", "BIOHUB_DIVCLF_PRUNE_THRESHOLD": "0.05"},
        patches=[(11, "divclf3_et.py")], replaces={}),
    # best private score: cell-state ILP (alpha 0.8, beta 0, gamma 1, rho 0.5) + readd + classifier mix
    "best_private_readd_mix": dict(
        slug="biohub-exp-readd-mix",
        overrides={"BIOHUB_CS_ALPHA": "0.8", "BIOHUB_CS_BETA": "0", "BIOHUB_CS_GAMMA": "1", "BIOHUB_CS_RHO": "0.5",
                   "BIOHUB_REPAIR_DEADLINE_S": "36000", "BIOHUB_DIVCLF_THRESHOLD": "0.4", "BIOHUB_DIVCLF_PRUNE_THRESHOLD": "0.05"},
        patches=[(3, "csilp.py"), (11, "divclf3_mix.py")], replaces={5: ROOT / "sub/replaces/csilp_pp.py"}),
}


def build(cfg):
    nb = json.load(open(BASE / "biohub-0-953-lb-original.ipynb"))
    reps = {}
    for ci, path in cfg["replaces"].items():
        ns = {}
        exec(Path(path).read_text(), ns)
        reps[ci] = ns["REPLACEMENTS"]
    cells = []
    for i, c in enumerate(nb["cells"]):
        c = copy.deepcopy(c)
        if c["cell_type"] == "code" and i in reps:
            src = "".join(c["source"])
            for old, new in reps[i]:
                assert src.count(old) == 1, f"replace anchor count {src.count(old)} in cell {i}"
                src = src.replace(old, new)
            c["source"] = src
        if c["cell_type"] == "code" and i == 0:
            c["source"] = "".join(c["source"]) + "\n\n# ---- experiment overrides\n_EXP_OVERRIDES = " + json.dumps(cfg["overrides"]) + \
                "\nos.environ.update(_EXP_OVERRIDES)\nprint('EXP_OVERRIDES', _EXP_OVERRIDES)\n"
        if c["cell_type"] == "code" and i == 1:
            c["source"] = "".join(c["source"]).replace('raise RuntimeError(\n        "Configuration drift detected: "',
                                                       'print(\n        "Configuration drift (experiment override): "')
        cells.append(c)
        for ci, name in cfg["patches"]:
            if ci == i:
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                              "source": (CELLS / name).read_text()})
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb["cells"] = cells
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build/notebooks"))
    ap.add_argument("--owner", default="YOUR_KAGGLE_USER", help="your Kaggle user name (kernel id)")
    ap.add_argument("--weights-dataset", default="YOUR_KAGGLE_USER/biohub-divclf3-weights")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ok = True
    for name, cfg in NOTEBOOKS.items():
        nb = build(cfg)
        meta = json.load(open(BASE / "kernel-metadata.json"))
        meta.update(id=f"{a.owner}/{cfg['slug']}", title=cfg["slug"], code_file="notebook.ipynb", is_private=True)
        meta.pop("id_no", None)
        meta["dataset_sources"] = meta["dataset_sources"] + [a.weights_dataset]
        out = Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        json.dump(nb, open(out / "notebook.ipynb", "w"))
        json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=1)
        if a.check:
            ref = ROOT / "notebooks" / name
            same_nb = (out / "notebook.ipynb").read_bytes() == (ref / "notebook.ipynb").read_bytes()
            same_meta = json.load(open(out / "kernel-metadata.json")) == json.load(open(ref / "kernel-metadata.json"))
            print(f"{name}: notebook identical={same_nb} metadata identical={same_meta}")
            ok &= same_nb and same_meta
        else:
            print("wrote", out)
    if a.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

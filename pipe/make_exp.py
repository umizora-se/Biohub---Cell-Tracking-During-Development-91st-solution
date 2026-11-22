"""Create a local experiment of the public notebook pipeline.

usage: python make_exp.py EXP_NAME [--nb path.ipynb] [--set KEY=VAL ...] [--eval eval16.txt]
Creates pipe/exps/EXP_NAME/{kaggle/input/..., kaggle/working/, run.py} and prints the run command.
BIOHUB_* overrides are applied right after notebook cell 0 (so they beat the notebook's hardcoded values).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "pipe"
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--nb", default=str(ROOT / "nb/biohub-0-953-lb-original/biohub-0-953-lb-original.ipynb"))
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--eval", default=str(PIPE / "eval16.txt"), type=lambda x: str(Path(x).resolve()))
ap.add_argument("--reuse", default=None, help="exp name whose predictions/edge_cache are reused (skip model inference)")
ap.add_argument("--patch", nargs="*", default=[], help="python files exec'd after the given cell: CELL:path")
ap.add_argument("--replace", nargs="*", default=[], help="CELL:path to a python file defining REPLACEMENTS=[(old,new),...] applied to that cell source")
a = ap.parse_args()

exp = PIPE / "exps" / a.name
kin = exp / "kaggle" / "input"
comp = kin / "competitions" / "biohub-cell-tracking-during-development"
(comp / "test").mkdir(parents=True, exist_ok=True)
(exp / "kaggle" / "working").mkdir(parents=True, exist_ok=True)
if not (comp / "train").exists():
    (comp / "train").symlink_to(ROOT / "data/train")
for owner in (PIPE / "kaggle/input/datasets").iterdir():
    (kin / "datasets" / owner.name).mkdir(parents=True, exist_ok=True)
    for slug in owner.iterdir():
        link = kin / "datasets" / owner.name / slug.name
        if not link.exists():
            link.symlink_to(slug.resolve())
for n in Path(a.eval).read_text().split():
    p = comp / "test" / f"{n}.zarr"
    if not p.exists():
        p.symlink_to(ROOT / "data/train" / f"{n}.zarr")

nb = json.load(open(a.nb))
overrides = dict(kv.split("=", 1) for kv in a.set)
patches = {}
for p in a.patch:
    c, path = p.split(":", 1)
    patches.setdefault(int(c), []).append(Path(path).resolve().read_text())
replaces = {}
for r in a.replace:
    ci, path = r.split(":", 1)
    ns = {}; exec(Path(path).resolve().read_text(), ns)
    replaces.setdefault(int(ci), []).extend(ns["REPLACEMENTS"])
cells = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    for old, new in replaces.get(i, []):
        assert src.count(old) == 1, f"replace anchor count {src.count(old)} in cell {i}: {old[:60]!r}"
        src = src.replace(old, new)
    src = src.replace('"/kaggle"', f'"{exp}/kaggle"').replace("/kaggle/", f"{exp}/kaggle/")
    if i == 0:
        src += "\n\n_LOCAL_OVERRIDES = " + json.dumps(overrides) + "\nos.environ.update(_LOCAL_OVERRIDES)\nprint('LOCAL_OVERRIDES', _LOCAL_OVERRIDES)\n"
    if a.reuse and i == 4:
        anchor = """    subprocess.run(
        predict_cmd,
        cwd=REPO_DIR,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )"""
        assert src.count(anchor) == 1, "reuse anchor"
        src_w = PIPE / "exps" / a.reuse / "kaggle" / "working"
        src = src.replace(anchor, f"""    import shutil as _sh
    _sh.copytree("{src_w}/tracking_repo/predictions", REPO_DIR / "predictions", dirs_exist_ok=True)
    _sh.copytree("{src_w}/edge_cache", "{exp}/kaggle/working/edge_cache", dirs_exist_ok=True)
    _keep = set(Path("{a.eval}").read_text().split())
    for _p in (REPO_DIR / "predictions").rglob("*.geff"):
        if _p.stem not in _keep:
            _sh.rmtree(_p)
    print("REUSED predictions from {a.reuse} (kept", len(_keep), "datasets)")""")
    src = src.replace('raise RuntimeError(\n        "DeepCenter checkpoint checksum mismatch: "', 'print(\n        "DeepCenter checkpoint checksum mismatch (ignored locally): "')
    src = re.sub(r"raise RuntimeError\(\s*\n\s*\"Configuration drift detected: \"", 'print(\n        "Configuration drift (ignored locally): "', src)
    cells.append(f"# %% [cell {i}]\n{src}")
    for k, psrc in enumerate(patches.get(i, [])):
        psrc = psrc.replace('"/kaggle"', f'"{exp}/kaggle"').replace("/kaggle/", f"{exp}/kaggle/")
        cells.append(f"# %% [cell {i}.patch{k}]\n{psrc}")
(exp / "run.py").write_text("\n\n".join(cells))
(exp / "config.json").write_text(json.dumps(dict(nb=a.nb, overrides=overrides, patches=a.patch, replaces=a.replace, eval=a.eval), indent=1))
print(f"cd {exp}/kaggle/working && {sys.executable} {PIPE}/run_cells.py {exp}/run.py > {exp}/run.log 2>&1")

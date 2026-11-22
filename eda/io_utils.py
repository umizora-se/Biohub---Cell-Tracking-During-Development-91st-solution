"""Minimal loaders for competition data (image zarr + geff tracks)."""
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

DATA = Path(__file__).resolve().parents[1] / "data"
SCALE = np.array([1.625, 0.40625, 0.40625])  # z, y, x (um / voxel)


def find_root():
    """Prefer fully extracted data, fall back to per-file downloads."""
    for p in (DATA, DATA / "single"):
        if (p / "train").exists():
            return p
    raise FileNotFoundError(DATA)


def load_geff(path):
    """Return (nodes DataFrame[id,t,z,y,x], edges DataFrame[src,dst], attrs)."""
    g = zarr.open_group(str(path), mode="r")
    ids = g["nodes/ids"][:]
    nodes = pd.DataFrame({"id": ids})
    for k in ("t", "z", "y", "x"):
        nodes[k] = g[f"nodes/props/{k}/values"][:]
    e = g["edges/ids"][:]
    edges = pd.DataFrame(e.astype(np.int64), columns=["src", "dst"])
    return nodes, edges, dict(g.attrs)["geff"]


def load_image(path):
    return zarr.open_array(str(Path(path) / "0"), mode="r")

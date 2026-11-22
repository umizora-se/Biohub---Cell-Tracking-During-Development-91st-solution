"""3D cell-state classifier (continue / parent / daughter) on 3-frame patches. Small ResNet, self-contained
(this file is also embedded in the Kaggle notebook for inference).

train:  python mitosis_cnn.py train --fold K --exclude names.txt --out runs/cnn_fK.pt   (K=-1 -> all data)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PZ, PY, PX = 7, 32, 32


class Block(nn.Module):
    def __init__(self, cin, cout, stride=(1, 2, 2)):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.sc = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm3d(cout)) if (cin != cout or stride != (1, 1, 1)) else nn.Identity()

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))


class MitosisNet(nn.Module):
    def __init__(self, in_ch=3, width=24, n_cls=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, width, 3, padding=1, bias=False), nn.BatchNorm3d(width), nn.ReLU())
        self.l1 = Block(width, width, (1, 1, 1))
        self.l2 = Block(width, width * 2, (1, 2, 2))
        self.l3 = Block(width * 2, width * 4, (2, 2, 2))
        self.l4 = Block(width * 4, width * 4, (1, 2, 2))
        self.head = nn.Linear(width * 4, n_cls)

    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        return self.head(x.mean((2, 3, 4)))


def augment(x, rng):
    # x: (B,3,PZ,PY,PX) tensor on device
    if rng.random() < 0.5:
        x = x.flip(-1)
    if rng.random() < 0.5:
        x = x.flip(-2)
    if rng.random() < 0.5:
        x = x.transpose(-1, -2)
    if rng.random() < 0.5:
        x = x.flip(-3)
    scale = torch.tensor(rng.uniform(0.8, 1.25, size=(x.shape[0], 1, 1, 1, 1)), dtype=x.dtype, device=x.device)
    return x * scale


@torch.no_grad()
def predict(model, X, device, bs=256, tta=True):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(np.asarray(X[i:i + bs], dtype=np.float32), device=device)
        logits = model(xb)
        if tta:
            logits = logits + model(xb.flip(-1)) + model(xb.flip(-2)) + model(xb.transpose(-1, -2))
            logits = logits / 4
        out.append(F.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


def train(a):
    d = np.load(Path(__file__).parent / "mitosis_patches.npz")
    X, y, soft, ds = d["X"], d["y"], d["soft"], d["ds"]
    excl = set(Path(a.exclude).read_text().split()) if a.exclude else set()
    m = ~np.isin(ds, list(excl))
    X, y, soft = X[m], y[m], soft[m]
    print("train patches", len(y), np.bincount(y), "excluded ds", len(excl), flush=True)
    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = MitosisNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * (len(y) // a.bs + 1))
    cw = torch.tensor([1.0, 4.0, 3.0], device=device)
    w_soft = torch.tensor(np.where(soft == 1, 0.5, 1.0), dtype=torch.float32)
    # oversample positives
    p = np.where(y == 0, 1.0, 6.0); p /= p.sum()
    for ep in range(a.epochs):
        model.train()
        idx = rng.choice(len(y), size=len(y), p=p)
        tot = 0
        for i in range(0, len(idx), a.bs):
            b = idx[i:i + a.bs]
            xb = torch.as_tensor(X[b].astype(np.float32), device=device)
            xb = augment(xb, rng)
            yb = torch.as_tensor(y[b], device=device)
            loss = (F.cross_entropy(model(xb), yb, weight=cw, reduction="none") * w_soft[b].to(device)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(idx):.4f}", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.half() for k, v in model.state_dict().items()}, "arch": "MitosisNet", "width": 24}, out)
    print("saved", out)


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    m = MitosisNet(width=ck.get("width", 24)).to(device)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)

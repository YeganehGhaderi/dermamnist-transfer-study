"""Run every saved checkpoint on the TEST set once; save logits and metrics.

    python scripts/evaluate_test.py
"""
import os

import numpy as np
import pandas as pd
import torch

from derma.data import CLASS_NAMES, DermaDataset, build_transforms
from derma.metrics import compute_metrics
from derma.models import build_model
from derma.utils import get_device

RUNS = "results/runs"
DATA = "data"
device = get_device()


def load_split(split):
    ds = DermaDataset(split, 64, DATA, build_transforms(False))   # False = no augmentation
    x = torch.stack([ds[i][0] for i in range(len(ds))])           # all images as one tensor
    return x, np.asarray(ds.labels).reshape(-1)


def make_model(name):
    try:
        return build_model(name, load_weights=False)   # weights come from our checkpoint
    except TypeError:
        return build_model(name)


def load_run(run):
    ck = torch.load(f"{RUNS}/{run}/best.pt", map_location="cpu")
    model = make_model(ck["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()                     # eval() is essential


@torch.no_grad()
def get_logits(model, x):
    out = [model(x[i:i + 256].to(device)).float().cpu() for i in range(0, len(x), 256)]
    return torch.cat(out).numpy()


x_val, y_val = load_split("val")
x_test, y_test = load_split("test")
print("val images:", len(y_val), "| test images:", len(y_test))

runs = [n for n in sorted(os.listdir(RUNS))
        if n.startswith(("expB_", "expC_", "expD_")) and "_frac" not in n
        and os.path.exists(f"{RUNS}/{n}/best.pt")]
print(len(runs), "runs found (expected 36)")

# ---- safety check on VALIDATION (not test) ----
saved = np.load(f"{RUNS}/{runs[0]}/val_logits.npz")
mine = compute_metrics(get_logits(load_run(runs[0]), x_val), y_val)["macro_f1"]
ref = compute_metrics(saved["logits"], saved["labels"])["macro_f1"]
print(f"pipeline check on val: now {mine:.4f} vs saved {ref:.4f}")
assert abs(mine - ref) < 0.005, "pipeline does not reproduce validation results: stop and tell Claude"

# ---- the real evaluation ----
rows = []
for run in runs:
    logits = get_logits(load_run(run), x_test)
    np.savez(f"{RUNS}/{run}/test_logits.npz", logits=logits, labels=y_test)
    m = compute_metrics(logits, y_test)
    pred = logits.argmax(1)
    recall = {f"recall_{c}": float((pred[y_test == k] == k).mean()) for k, c in enumerate(CLASS_NAMES)}
    name, seed = run.rsplit("_seed", 1)
    rows.append(dict(name=name, seed=int(seed), **m, **recall))
    print(f"{run:42s} test macro-F1 {m['macro_f1']:.3f}")

pd.DataFrame(rows).to_csv("results/test_per_run.csv", index=False)
print("saved results/test_per_run.csv")
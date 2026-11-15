"""Calibration on the TEST set: T is fitted on validation logits only.

    python scripts/test_calibration.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derma.calibration import TemperatureScaler, plot_reliability
from derma.data import CLASS_NAMES
from derma.metrics import softmax

R = "results/runs"
RARE = [list(CLASS_NAMES).index(c) for c in ("akiec", "bcc", "df", "vasc")]
B = ["expB_ce", "expB_weighted_ce", "expB_focal", "expB_oversample"]
D = ["expD_smallcnn", "expD_resnet18_scratch", "expD_resnet18_pretrained"]


def ece(probs, labels, n_bins=15):
    """Expected calibration error: average gap between confidence and accuracy."""
    conf, correct = probs.max(1), probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def rare_gap(probs, labels):
    """Mean confidence minus accuracy, only for predictions of the rare classes."""
    pred = probs.argmax(1)
    m = np.isin(pred, RARE)
    return float(probs.max(1)[m].mean() - (pred[m] == labels[m]).mean())


def run_calibration(run):
    v = np.load(f"{R}/{run}/val_logits.npz")
    t = np.load(f"{R}/{run}/test_logits.npz")
    sc = TemperatureScaler()
    T = sc.fit(v["logits"], v["labels"])                 # T from validation ONLY
    p0, p1, y = softmax(t["logits"]), softmax(sc.transform(t["logits"])), t["labels"]
    res = dict(T=T, ece_before=ece(p0, y), ece_after=ece(p1, y),
               gap_before=rare_gap(p0, y), gap_after=rare_gap(p1, y))
    return res, (p0, p1, y)


rows = []
for name in B + D:
    for seed in (0, 1, 2):
        res, _ = run_calibration(f"{name}_seed{seed}")
        rows.append(dict(name=name, seed=seed, **res))
df = pd.DataFrame(rows)
df.to_csv("results/test_calibration_per_run.csv", index=False)

cols = ["T", "ece_before", "ece_after", "gap_before", "gap_after"]
g = df.groupby("name")[cols]
mean, std = g.mean(), g.std()
pretty = pd.DataFrame({c: [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])] for c in cols},
                      index=mean.index).loc[B + D]
pretty.to_csv("results/test_calibration.csv", encoding="utf-8-sig")
print("\nCalibration on the TEST set (mean ± std over 3 seeds)")
print(pretty.to_string())

# ---- reliability diagrams: seed 0 of each imbalance strategy ----
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for col, n in enumerate(B):
    _, (p0, p1, y) = run_calibration(f"{n}_seed0")
    plot_reliability(axes[0, col], p0, y, title=f"{n[5:]}: before")
    plot_reliability(axes[1, col], p1, y, title=f"{n[5:]}: after")
plt.tight_layout()
plt.savefig("figures/test_reliability_imbalance.png", dpi=150)
print("saved figures/test_reliability_imbalance.png")
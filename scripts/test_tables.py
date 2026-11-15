"""Tables and figures on the test set (mean +- std over seeds).

    python scripts/test_tables.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from derma.data import CLASS_NAMES

os.makedirs("figures", exist_ok=True)
CLASSES = list(CLASS_NAMES)
df = pd.read_csv("results/test_per_run.csv")
METRICS = ["macro_f1", "balanced_acc", "auc_macro", "accuracy"]
RECALLS = [f"recall_{c}" for c in CLASSES]

TABLES = {
    "table1_models": ["expD_smallcnn", "expD_resnet18_scratch", "expD_resnet18_pretrained"],
    "table2_imbalance": ["expB_ce", "expB_weighted_ce", "expB_focal", "expB_oversample"],
    "table3_augmentation": ["expC_0_none", "expC_1_flips", "expC_2_flips_rot",
                            "expC_3_flips_rot_color", "expC_4_full"],
}


def mean_std(names, cols):
    g = df[df.name.isin(names)].groupby("name")[cols]
    return g.mean().loc[names], g.std().loc[names]


for key, names in TABLES.items():
    mean, std = mean_std(names, METRICS + RECALLS)
    pretty = mean.copy().astype(object)
    for c in mean.columns:
        pretty[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    pretty.to_csv(f"results/test_{key}.csv", encoding="utf-8-sig")
    print(f"\n=== {key} (TEST set, mean ± std over 3 seeds) ===")
    print(pretty[METRICS].to_string())
    print(pretty[RECALLS].to_string())

# ---- confusion matrix of the final model (mean of 3 seeds, rows sum to 1) ----
cms = []
for s in (0, 1, 2):
    z = np.load(f"results/runs/expD_resnet18_pretrained_seed{s}/test_logits.npz")
    cms.append(confusion_matrix(z["labels"], z["logits"].argmax(1),
                                labels=list(range(7)), normalize="true"))
cm = np.mean(cms, axis=0)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > 0.5 else "black")
ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
ax.set_title("Pretrained ResNet-18, test set\n(row-normalised, mean of 3 seeds)")
fig.colorbar(im); plt.tight_layout()
plt.savefig("figures/test_confusion_pretrained.png", dpi=150); plt.close()
mel = CLASSES.index("mel")
print("\nwhere true melanoma images go:", dict(zip(CLASSES, cm[mel].round(2))))

# ---- recall per class for each imbalance strategy ----
names = TABLES["table2_imbalance"]
mean, std = mean_std(names, RECALLS)
x, w = np.arange(7), 0.2
fig, ax = plt.subplots(figsize=(10, 4.5))
for k, n in enumerate(names):
    ax.bar(x + (k - 1.5) * w, mean.loc[n].values, w, yerr=std.loc[n].values,
           capsize=2, label=n.replace("expB_", ""))
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1); ax.set_ylabel("recall (test set)")
ax.set_title("Per-class recall by imbalance strategy (mean ± std, 3 seeds)")
ax.legend(); plt.tight_layout()
plt.savefig("figures/test_recall_by_imbalance.png", dpi=150); plt.close()
print("saved figures/test_confusion_pretrained.png and figures/test_recall_by_imbalance.png")
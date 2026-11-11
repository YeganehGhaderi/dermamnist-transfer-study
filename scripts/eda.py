"""Class distribution table + Figure 1 (bar chart) + Figure 2 (sample grid).

    python scripts/eda.py --data_root data
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derma.data import CLASS_NAMES, DermaDataset, class_counts, compute_class_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    splits = {s: DermaDataset(s, args.size, args.data_root) for s in ["train", "val", "test"]}

    # ----- counts table -----
    table = pd.DataFrame({s: class_counts(ds.labels) for s, ds in splits.items()}, index=CLASS_NAMES)
    table["train_%"] = (100 * table["train"] / table["train"].sum()).round(2)
    table["class_weight"] = compute_class_weights(splits["train"].labels).round(3)
    table.loc["total"] = table.sum(numeric_only=True)
    table.loc["total", ["train_%", "class_weight"]] = np.nan
    print(table)
    table.to_csv("results/class_counts.csv")

    # ----- Figure 1: class distribution of the training set -----
    counts = class_counts(splits["train"].labels)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(CLASS_NAMES, counts, color="#4C72B0")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({100 * c / counts.sum():.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("training images")
    ax.set_title("DermaMNIST: class distribution (train split)")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig1_class_distribution.png", dpi=200)

    # ----- Figure 2: 6 random examples per class -----
    rng = np.random.default_rng(0)
    ds, per_class = splits["train"], 6
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(per_class * 1.2, len(CLASS_NAMES) * 1.25))
    for c, name in enumerate(CLASS_NAMES):
        idx = rng.choice(np.where(ds.labels == c)[0], per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(ds.images[i])
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("DermaMNIST examples (64x64)")
    fig.tight_layout()
    fig.savefig("figures/fig2_examples.png", dpi=200)
    print("saved figures/fig1_class_distribution.png and figures/fig2_examples.png")


if __name__ == "__main__":
    main()

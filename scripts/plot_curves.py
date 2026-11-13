"""Plot loss and val Macro-F1 curves for several runs.

    python scripts/plot_curves.py results/runs/smallcnn_seed0 results/runs/resnet18_scratch_seed0 \
        results/runs/resnet18_pretrained_seed0 --out figures/expA_curves.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="figures/curves.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in args.runs:
        h = pd.read_csv(os.path.join(run, "history.csv"))
        name = os.path.basename(run.rstrip("/"))
        line, = axes[0].plot(h["epoch"], h["train_loss"], label=f"{name} train")
        axes[0].plot(h["epoch"], h["val_loss"], "--", color=line.get_color(), label=f"{name} val")
        axes[1].plot(h["epoch"], h["val_macro_f1"], color=line.get_color(), label=name)
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss (solid: train loss, dashed: val cross-entropy)")
    axes[1].set(xlabel="epoch", ylabel="Macro-F1", title="Validation Macro-F1")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()

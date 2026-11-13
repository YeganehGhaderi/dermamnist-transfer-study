"""Mean ± std over seeds for a group of runs, plus per-class recall.

    python scripts/summarize.py --runs_dir /content/drive/MyDrive/derma/runs \
        --names expB_ce expB_weighted_ce expB_focal expB_oversample --out results/expB
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from derma.data import CLASS_NAMES

METRICS = ["val_macro_f1", "val_balanced_acc", "val_auc_macro", "val_accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="prefix for the csv files, e.g. results/expB")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    main_rows, recall_rows = [], []
    for name in args.names:
        run_dirs = sorted(glob.glob(os.path.join(args.runs_dir, f"{name}_seed*")))
        run_dirs = [d for d in run_dirs if os.path.exists(os.path.join(d, "summary.json"))]
        if not run_dirs:
            print(f"warning: no finished runs for {name}")
            continue
        summaries = [json.load(open(os.path.join(d, "summary.json"))) for d in run_dirs]
        row = {"name": name, "seeds": len(summaries),
               "best_epoch": np.mean([s["best_epoch"] for s in summaries])}
        for m in METRICS:
            vals = np.array([s[m] for s in summaries])
            row[m] = vals.mean()
            row[m + "_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        main_rows.append(row)

        recalls = []
        for d in run_dirs:   # recall of every class = "what fraction of class c did we find?"
            z = np.load(os.path.join(d, "val_logits.npz"))
            recalls.append(recall_score(z["labels"], z["logits"].argmax(1),
                                        labels=list(range(len(CLASS_NAMES))), average=None, zero_division=0))
        recall_rows.append({"name": name, **dict(zip(CLASS_NAMES, np.mean(recalls, axis=0)))})

    main_df = pd.DataFrame(main_rows).set_index("name")
    recall_df = pd.DataFrame(recall_rows).set_index("name")
    main_df.round(4).to_csv(args.out + "_summary.csv")
    recall_df.round(4).to_csv(args.out + "_recall.csv")

    pretty = pd.DataFrame({m.replace("val_", ""): [f"{r[m]:.3f} ± {r[m + '_std']:.3f}" for _, r in main_df.iterrows()]
                           for m in METRICS}, index=main_df.index)
    pretty.insert(0, "seeds", main_df["seeds"].astype(int))
    pretty["best_epoch"] = main_df["best_epoch"].round(1)
    print("\nMean ± std over seeds (validation set)\n")
    print(pretty.to_string())
    print("\nRecall per class (mean over seeds)\n")
    print(recall_df.round(3).to_string())


if __name__ == "__main__":
    main()

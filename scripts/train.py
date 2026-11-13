"""Train one model with one seed.

Example:
    python scripts/train.py --config configs/smallcnn.yaml --seed 0 --out_dir results/runs
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from derma.data import make_loaders
from derma.engine import EarlyStopping, evaluate, train_one_epoch
from derma.losses import build_criterion
from derma.metrics import compute_metrics
from derma.models import build_model, count_parameters, param_groups
from derma.utils import get_device, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="results/runs")
    parser.add_argument("--data_root", default=None, help="overrides data_root in the config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set loss=focal augment=full")
    parser.add_argument("--name", default=None, help="overrides the run name in the config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root
    for item in args.set:                       # "loss=focal" -> cfg["loss"] = "focal"
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)        # turns "60" into 60 and "0.1" into 0.1
    if args.name:
        cfg["name"] = args.name
    cfg["seed"] = args.seed

    run_dir = os.path.join(args.out_dir, f"{cfg['name']}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device} | run: {run_dir}")

    # ----- data, model, loss, optimizer, scheduler -----
    train_loader, val_loader = make_loaders(cfg, args.seed)
    model = build_model(cfg["model"]).to(device)
    print(f"parameters: {count_parameters(model):,}")

    # training loss: plain / weighted cross-entropy or Focal Loss, depending on the config
    criterion = build_criterion(cfg, train_loader.dataset.labels, device)
    # validation loss is ALWAYS plain cross-entropy, so runs with different training losses compare fairly
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        param_groups(model, cfg["model"], cfg["lr"], cfg.get("lr_backbone", cfg["lr"])),
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper = EarlyStopping(patience=cfg["patience"], mode="max")

    # ----- training loop -----
    history_path = os.path.join(run_dir, "history.csv")
    fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_bal_acc", "val_acc", "lr", "seconds"]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[-1]["lr"]   # lr of the head (the larger one)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_logits, val_labels = evaluate(model, val_loader, eval_criterion, device)
        m = compute_metrics(val_logits, val_labels)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                   val_macro_f1=round(m["macro_f1"], 5), val_bal_acc=round(m["balanced_acc"], 5),
                   val_acc=round(m["accuracy"], 5), lr=lr_now, seconds=round(time.time() - t0, 1))
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        improved = stopper.step(m["macro_f1"])
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": m["macro_f1"],
                        "config": cfg}, os.path.join(run_dir, "best.pt"))
        print(f"ep {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"F1 {m['macro_f1']:.4f} | balAcc {m['balanced_acc']:.4f}{'  *' if improved else ''}")
        if stopper.should_stop:
            print(f"early stopping at epoch {epoch} (best F1 {stopper.best:.4f})")
            break

    # ----- reload the best checkpoint, save its val logits and a summary -----
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    _, val_logits, val_labels = evaluate(model, val_loader, eval_criterion, device)
    np.savez(os.path.join(run_dir, "val_logits.npz"), logits=val_logits, labels=val_labels)
    summary = {"best_epoch": ckpt["epoch"], "epochs_run": epoch,
               "params": count_parameters(model), **{f"val_{k}": v for k, v in
                                                    compute_metrics(val_logits, val_labels).items()}}
    save_json(summary, os.path.join(run_dir, "summary.json"))
    print("summary:", summary)


if __name__ == "__main__":
    main()

"""Run several configs x several seeds one after another.

Runs that already have a summary.json are skipped, so after a Colab disconnect you can simply run
the same command again and it continues where it stopped.

    python scripts/sweep.py --configs configs/expB_*.yaml --seeds 0 1 2 \
        --data_root /content/drive/MyDrive/derma/data --out_dir /content/drive/MyDrive/derma/runs
    # extra overrides go to every run:
    python scripts/sweep.py --configs configs/expC_*.yaml --seeds 0 1 2 --set loss=focal ...
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--suffix", default="", help="added to every run name, e.g. _frac10")
    args = parser.parse_args()

    # Windows' cmd does not expand "configs/expB_*.yaml" by itself, so we do it here
    config_paths = sorted({p for pattern in args.configs for p in (glob.glob(pattern) or [pattern])})
    jobs = []
    for cfg_path in config_paths:
        with open(cfg_path) as f:
            name = yaml.safe_load(f)["name"] + args.suffix
        for seed in args.seeds:
            jobs.append((cfg_path, name, seed))

    t_start = time.time()
    for i, (cfg_path, name, seed) in enumerate(jobs, 1):
        run_dir = os.path.join(args.out_dir, f"{name}_seed{seed}")
        if os.path.exists(os.path.join(run_dir, "summary.json")):
            print(f"[{i}/{len(jobs)}] skip {name} seed {seed} (already done)")
            continue
        print(f"\n[{i}/{len(jobs)}] {name} seed {seed}   ({(time.time() - t_start) / 60:.1f} min so far)",
              flush=True)
        cmd = [sys.executable, "scripts/train.py", "--config", cfg_path, "--seed", str(seed),
               "--data_root", args.data_root, "--out_dir", args.out_dir, "--name", name]
        if args.set:
            cmd += ["--set", *args.set]
        subprocess.run(cmd, check=True)
    print(f"\nall {len(jobs)} runs done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()

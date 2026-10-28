"""Step 6A: train the learned path initializer on the P3 v1 train split (on the fly, variable G), then calibrate the
detection threshold on the validation split and freeze it into the checkpoint (never tuned on test).

    python scripts/12_train_path_init.py --samples 400000 --batch 128
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import ROOT
from positioning.datasets.generator import simulate_split_sample
from positioning.datasets.splits import load_splits
from positioning.evaluation.benchmark import G_GRID
from positioning.models.data import PathHeatmapDataset
from positioning.models.path_init import PathInitNet, decode, network_input, path_loss
from positioning.system.simulator import subsample_beams

import os
from pathlib import Path

# dataset / output selectable for the v2 round (v1R = same recipe on train_v2); defaults reproduce v1 exactly
OUT = Path(os.environ.get("MMW_OUT", ROOT / "outputs" / "models" / "path_init"))
SPEC_PATH = Path(os.environ.get("MMW_DATASET", ROOT / "configs" / "datasets" / "p3_v1.yaml"))
TRAIN_SPLIT = os.environ.get("MMW_TRAIN_SPLIT", "train")
VAL_SPLIT = os.environ.get("MMW_VAL_SPLIT", "val")


def calibrate_threshold(net, spec, val, n=1200):
    """Threshold maximising P(K_hat = K) on validation samples (all G of the grid, cycled)."""
    net.eval()
    outs, Ks, cfgs = [], [], []
    with torch.no_grad():
        for i, seed in enumerate(val.seeds()[:n]):
            s, _, _ = simulate_split_sample(spec, val, int(seed), with_bounds=False)
            s = subsample_beams(s, G_GRID[i % len(G_GRID)])
            outs.append(net(torch.from_numpy(network_input(s.y, s.context))[None])[0].numpy())
            Ks.append(s.gt.num_paths)
            cfgs.append(s.context.config)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    acc = [float(np.mean([len(decode(o, c, t)) == k for o, c, k in zip(outs, cfgs, Ks)])) for t in grid]
    net.train()
    return float(grid[int(np.argmax(acc))]), dict(zip(map(float, grid), acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--threads", type=int, default=28)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    spec, splits = load_splits(SPEC_PATH)
    train = next(s for s in splits if s.name == TRAIN_SPLIT)
    val = next(s for s in splits if s.name == VAL_SPLIT)
    # train seeds disjoint from those of the direct CNN (first 800k) and LAMP (last 30k)
    seeds = train.seeds()[800_000:800_000 + args.samples] if args.samples <= 170_000 else \
        np.r_[train.seeds()[800_000:970_000], train.seeds()[:args.samples - 170_000]]
    dl = DataLoader(PathHeatmapDataset(spec, train, seeds), batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
    net = PathInitNet(width=args.width)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=len(dl), pct_start=max(0.05, 3 / len(dl)))
    OUT.mkdir(parents=True, exist_ok=True)
    log, t0, run = [], time.time(), []
    for step, (x, h, r, m) in enumerate(dl, 1):
        loss = path_loss(net(x), h, r, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 100 == 0 or step == len(dl):
            log.append({"step": step, "loss_avg100": float(np.mean(run[-100:])), "elapsed_s": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
            torch.save({"state_dict": net.state_dict(), "model_kwargs": {"width": args.width}, "args": vars(args),
                        "log": log, "dataset": SPEC_PATH.stem}, OUT / "model.pt")
    thr, curve = calibrate_threshold(net, spec, val)
    print(f"calibrated threshold {thr} (validation P(K_hat=K) = {curve[thr]:.3f})")
    torch.save({"state_dict": net.state_dict(), "model_kwargs": {"width": args.width}, "args": vars(args), "log": log,
                "dataset": SPEC_PATH.stem, "threshold": thr, "threshold_curve": curve, "threshold_split": "val"},
               OUT / "model.pt")
    (OUT / "train_log.json").write_text(json.dumps({"log": log, "threshold": thr, "curve": curve}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

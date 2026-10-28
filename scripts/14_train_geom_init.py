"""Step 6B-1: train the learned OLOS geometric initializer (geometry-only data from the P3 v1 prior).

    python scripts/14_train_geom_init.py --samples 1500000 --epochs 6
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
import torch

from _bootstrap import ROOT
from pathlib import Path
from positioning.datasets.splits import load_splits
from positioning.models.geom_init import GeomInitNet, geom_loss, perturb, sample_geometries

import os

# dataset / output selectable for the v2 round (v1R = same recipe on train_v2); defaults reproduce v1 exactly.
# Passed through the environment so that forkserver workers see the same values.
OUT = Path(os.environ.get("MMW_OUT", ROOT / "outputs" / "models" / "geom_init"))
SPEC = Path(os.environ.get("MMW_DATASET", ROOT / "configs" / "datasets" / "p3_v1.yaml"))
TRAIN_SPLIT = os.environ.get("MMW_TRAIN_SPLIT", "train")


def _chunk(seeds):
    """Geometry chunk. A seed for which the frozen half-plane rejection sampler finds no scatterer behind the MS
    array (possible, very rarely, in the wider p3_v2 box) is skipped; the number of skipped seeds is reported."""
    spec, splits = load_splits(SPEC)
    split = next(s for s in splits if s.name == TRAIN_SPLIT)
    try:
        return sample_geometries(spec, split, seeds)
    except RuntimeError:
        parts = []
        for s in seeds:
            try:
                parts.append(sample_geometries(spec, split, [s]))
            except RuntimeError:
                print(f"skipped geometry seed {int(s)} (half-plane sampler)", flush=True)
        return [np.concatenate([q[i] for q in parts]) for i in range(5)]


def build(seeds, workers):
    chunks = np.array_split(seeds, workers * 4)
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_chunk, chunks))
    return [np.concatenate([p[i] for p in parts]) for i in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1_500_000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=46)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    _, splits = load_splits(SPEC)
    train = next(s for s in splits if s.name == TRAIN_SPLIT)
    t0 = time.time()
    # geometry seeds: an own sub-range of the train split (disjoint from the seeds used by the other models)
    X, S, M, P, A = build(train.seeds()[-args.samples - 30_000:-30_000], args.workers)
    val = build(train.seeds()[-30_000:-20_000], args.workers)
    print(f"data: {X.shape[0]} train / {val[0].shape[0]} held-out geometries in {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(0)
    net = GeomInitNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.epochs * (X.shape[0] // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    Xv, Mv = perturb(val[0], val[2], np.random.default_rng(1))
    log = []
    step = 0
    for ep in range(args.epochs):
        perm = rng.permutation(X.shape[0])
        for i in range(0, X.shape[0] - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            xb, mb = perturb(X[idx], M[idx], rng)
            pose, per = net(torch.from_numpy(xb), torch.from_numpy(mb))
            loss = geom_loss(pose, per, torch.from_numpy(P[idx]), torch.from_numpy(A[idx]), torch.from_numpy(S[idx]),
                             torch.from_numpy(mb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        with torch.no_grad():
            pose, per = net(torch.from_numpy(Xv), torch.from_numpy(Mv))
            pe = np.linalg.norm((pose[:, :2].numpy() - val[3]) * np.array([1.0, 0.15]), axis=1)
            ae = np.abs(np.angle(np.exp(1j * (np.arctan2(pose[:, 3], pose[:, 2]).numpy() - val[4]))))
        log.append({"epoch": ep + 1, "loss": loss.item(), "heldout_median_p_err_m": float(np.median(pe)),
                    "heldout_median_alpha_err": float(np.median(ae)), "elapsed_s": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "model_kwargs": {"width": 128}, "args": vars(args), "log": log,
                "dataset": f"{SPEC.stem} (geometry only)"}, OUT / "model.pt")
    (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

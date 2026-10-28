"""Generate the stored splits of a dataset spec (master plan P3). train is on-the-fly and skipped unless --include-train.

    python scripts/04_generate_dataset.py --splits val test_iid --limit 200
"""
import argparse
import time

import _bootstrap  # noqa: F401

from _bootstrap import ROOT
from positioning.datasets.generator import generate_split, load_split
from positioning.datasets.splits import load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(ROOT / "configs" / "datasets" / "p3_v1.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "datasets" / "p3_v1"))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap samples per split (smoke runs)")
    ap.add_argument("--include-train", action="store_true")
    ap.add_argument("--no-bounds", action="store_true")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg_path, splits = load_splits(args.spec)
    import yaml
    version = (yaml.safe_load(open(args.spec)) or {}).get("dataset_version")
    for sp in splits:
        if args.splits and sp.name not in args.splits:
            continue
        if not sp.stored and not args.include_train:
            print(f"skip {sp.name} (on-the-fly split)")
            continue
        if args.limit:
            sp.num_samples = min(sp.num_samples, args.limit)
        t = time.time()
        path = generate_split(cfg_path, sp, args.out, args.shard_size, not args.no_bounds, args.workers,
                              spec_path=str(args.spec), dataset_version=version)
        d = load_split(path)
        print(f"{sp.name}: {d['y'].shape[0]} samples, y {d['y'].shape} {d['y'].dtype}, "
              f"conditions {sorted(set(d['condition'].tolist()))}, {time.time() - t:.1f} s -> {path}")


if __name__ == "__main__":
    main()

"""PAPER-FILL runner: the per-observation evaluation of scripts/151_final_eval.py (same sample draw, same scoring, same
record format) for the new arms of pf_arms.py. Timing protocol of the paper: 8 worker processes, 1 computation thread each.

    python scripts/paper_fill/run_pf.py --set fconf --cell latneg_olos --G 16,32 --n 3000 --arms pf_full,pf_nosupp,...
    python scripts/paper_fill/run_pf.py --set fval  --cell iid         --G 16,32 --n 900  --arms ...

Cells: latneg_olos = region latneg, condition olos; iid = region iid, no condition; standard SNR uniform [-10, 10] dB.
Records: results/paper_fill/records/<set>_<cell>__<tag>.jsonl
"""
import argparse
import importlib.util
import multiprocessing
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(HERE)]

spec = importlib.util.spec_from_file_location("final_eval", ROOT / "scripts" / "151_final_eval.py")
FE = importlib.util.module_from_spec(spec)
sys.modules["final_eval"] = FE
spec.loader.exec_module(FE)
import pf_arms  # noqa: E402

_PF = {}


def _est(name):
    if name.startswith("pf_"):
        if name not in _PF:
            import torch
            torch.set_num_threads(1)
            _PF[name] = pf_arms.build(name)
        return _PF[name]
    return _ORIG_EST(name)


_ORIG_EST = FE._est
FE._est = _est
CELLS = {"latneg_olos": ("latneg", "olos"), "iid": ("iid", None)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["fconf", "fval"], required=True)
    ap.add_argument("--cell", choices=list(CELLS), required=True)
    ap.add_argument("--G", type=lambda s: tuple(int(v) for v in s.split(",")), default=(16, 32))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--arms", type=lambda s: s.split(","), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for arm in a.arms:
        FE.ARMS.setdefault(arm, {"kind": "pf"})
    region, cond = CELLS[a.cell]
    split = f"{a.set}_{a.cell}"
    out = ROOT / "results" / "paper_fill" / "records" / f"{split}__{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), out
    items = [(split, region, i, a.G, a.arms, False, 10 ** 9, cond) for i in range(a.n)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("fork")) as ex, open(out, "w") as fh:
        for recs in ex.map(FE.work, items, chunksize=1):
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    print(out.name, "done", f"{time.time() - t0:.0f}s", "workers", a.workers, flush=True)

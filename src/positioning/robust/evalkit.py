"""Evaluation helpers shared by the scripts of the program (ground truth is used here only, for scoring)."""
from __future__ import annotations

import numpy as np


def match(gt, est, Ts, nt, nr, use_aod=True):
    """As scripts/35_v3_diag.py::_match: |dtau| <= Ts and |d sin| <= one cell. gt / est = (tau, aod, aoa)."""
    from scipy.optimize import linear_sum_assignment
    if est is None or len(est[0]) == 0 or len(gt[0]) == 0:
        return {}
    d = np.abs(np.subtract.outer(gt[0], est[0])) / Ts
    ar = np.abs(np.subtract.outer(np.sin(gt[2]), np.sin(est[2]))) * nr / 2
    at = np.abs(np.subtract.outer(np.sin(gt[1]), np.sin(est[1]))) * nt / 2
    ok = (d <= 1.0) & (ar <= 1.0) & ((at <= 1.0) if use_aod else True)
    cost = np.where(ok, d + ar + (at if use_aod else 0), 1e6)
    r, c = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(r, c) if ok[i, j]}


def score(out, sample, t: float) -> dict:
    """Record of one estimate: err / ori (None = failed), K-hat, matched true paths, condition, loop counters, time."""
    cfg = sample.context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    gp = sample.gt.paths
    gt = (np.asarray(gp.delay), np.asarray(gp.aod), np.asarray(gp.aoa))
    fin = out is not None and bool(np.all(np.isfinite(out.position)))
    r = {"err": float(np.linalg.norm(out.position - sample.gt.position)) if fin else None,
         "ori": float(abs(np.angle(np.exp(1j * (out.orientation - sample.gt.orientation))))) if fin and np.isfinite(
             out.orientation) else None, "t": float(t), "khat": 0, "nm": 0, "nm_noaod": 0}
    ch = getattr(out, "channel", None) if out is not None else None
    if ch is not None and ch.num_paths:
        E = (np.asarray(ch.delay), np.asarray(ch.aod), np.asarray(ch.aoa))
        r.update({"khat": int(ch.num_paths), "nm": len(match(gt, E, Ts, nt, nr)),
                  "nm_noaod": len(match(gt, E, Ts, nt, nr, use_aod=False))})
    if out is not None:
        info = out.info
        v = info.get("paths") or info.get("loop") or {}
        r["acc"] = int(v.get("accepted", 0))
        r["rem"] = int(v.get("removed", 0))
        r["fwd"] = int(v.get("local_calls", v.get("fwd_entered", 0)) > 0)
        if "n0_used" in info:
            r["n0_used"] = float(info["n0_used"])
    return r


def bounds(sample):
    from ..bounds.peb_reb import compute_bounds
    try:
        b = compute_bounds(sample)
        return float(b.peb), float(b.reb)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

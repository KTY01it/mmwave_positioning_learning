"""Statistics shared by the reports of the program: failure indicators, summaries, paired bootstrap."""
from __future__ import annotations

import numpy as np


def err(r, arm):
    v = r["arms"][arm]["err"]
    return np.nan if v is None else v


def f_out(r, arm):
    return float(not (err(r, arm) <= 3 * r["peb"]))


def f_cat(r, arm):
    return float(not (err(r, arm) <= 1.0))


def boot(d, n=3000, seed=0):
    """(mean, lo, hi) in percentage points of a per-sample difference of indicators."""
    d = np.asarray(d, float)
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n, d.size))].mean(1) * 100
    return 100 * float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(R, a, b, f=f_out):
    return boot([f(r, a) - f(r, b) for r in R])


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def summary(R, arm):
    e = np.array([err(r, arm) for r in R])
    o = np.array([np.nan if r["arms"][arm]["ori"] is None else r["arms"][arm]["ori"] for r in R])
    A = [r["arms"][arm] for r in R]
    K = sum(r["K"] for r in R)
    t = np.array([a["t"] for a in A]) * 1e3
    return {"n": len(R), "p_out": 100 * float(np.mean([f_out(r, arm) for r in R])),
            "p_cat": 100 * float(np.mean([f_cat(r, arm) for r in R])), "fail": 100 * float(np.mean(~np.isfinite(e))),
            "rmse": float(np.sqrt(np.nanmean(e ** 2))), "median": float(np.nanmedian(e)),
            "ori_rmse": float(np.sqrt(np.nanmean(o ** 2))) if np.isfinite(o).any() else float("nan"),
            "recall": sum(a["nm"] for a in A) / max(K, 1), "precision": sum(a["nm"] for a in A) / max(sum(a["khat"] for a in A), 1),
            "p_k": float(np.mean([a["khat"] == r["K"] for a, r in zip(A, R)])),
            "acc": float(np.mean([a.get("acc", 0) for a in A])), "rem": float(np.mean([a.get("rem", 0) for a in A])),
            "fwd": 100 * float(np.mean([a.get("fwd", 0) for a in A])),
            "t_mean": float(t.mean()), "t_med": float(np.median(t)), "t_p95": float(np.percentile(t, 95))}


def line(name, s):
    return (f"{name:16s} {s['p_out']:6.2f} {s['p_cat']:6.2f} {s['fail']:5.1f} {s['rmse']:7.3f} {s['median']:8.4f} {s['ori_rmse']:7.3f} "
            f"{s['recall']:6.3f} {s['precision']:6.3f} {s['p_k']:6.3f} {s['acc']:6.3f} {s['rem']:6.3f} {s['fwd']:6.1f} "
            f"{s['t_mean']:7.0f} {s['t_med']:6.0f} {s['t_p95']:6.0f}")


HEAD = ("arm               P_out  P_cat fail%    RMSE   median  oriRMSE recall   prec P(K=K)  add/s  rem/s  fwd%   "
        "t_mean  t_med  t_p95")

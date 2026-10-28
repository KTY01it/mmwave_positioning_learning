"""Evaluation metrics (master plan P2.2-P2.3). Delta and epsilon are project-defined normalisations."""
from __future__ import annotations

import numpy as np

from ..system.geometry import wrap_angle


def angle_error(est, true):
    """d_angle = |wrap(est - true)|."""
    return np.abs(wrap_angle(np.asarray(est, float) - np.asarray(true, float)))


def position_error(est, true):
    return np.linalg.norm(np.asarray(est, float) - np.asarray(true, float), axis=-1)


def rmse(errors) -> float:
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def summary(errors) -> dict:
    e = np.asarray(errors, float)
    fin = e[np.isfinite(e)]
    if not fin.size:
        return {"rmse": np.nan, "mae": np.nan, "median": np.nan, "p90": np.nan, "n": 0, "n_failed": int(e.size)}
    return {"rmse": rmse(fin), "mae": float(np.mean(np.abs(fin))), "median": float(np.median(np.abs(fin))),
            "p90": float(np.percentile(np.abs(fin), 90)), "n": int(fin.size), "n_failed": int(e.size - fin.size)}


def delta_db(rmse_value: float, bound_rms: float) -> float:
    """Delta = 10 log10(RMSE^2 / bound^2) for a fixed geometry (P2.2, way 1)."""
    return float(10 * np.log10(rmse_value ** 2 / bound_rms ** 2))


def normalized_errors(errors, bounds):
    """epsilon_i = err_i^2 / bound_i^2 for random geometries (P2.2, way 2)."""
    return np.asarray(errors, float) ** 2 / np.asarray(bounds, float) ** 2


def mean_normalized_db(errors, bounds) -> float:
    return float(10 * np.log10(np.nanmean(normalized_errors(errors, bounds))))


def outlier_rate(errors, bounds, factor: float = 3.0) -> float:
    """P(||p_hat - p|| > factor * PEB)."""
    e, b = np.asarray(errors, float), np.asarray(bounds, float)
    return float(np.mean(~(e <= factor * b)))  # NaN (failure) counts as outlier


def snr_threshold(snrs, deltas, tol_db: float = 1.0, failure_rates=None, max_failure: float = 0.01) -> float:
    """min SNR such that Delta <= tol (and failure rate <= max_failure) for this and all higher SNRs (P2.3)."""
    snrs, deltas = np.asarray(snrs, float), np.asarray(deltas, float)
    fr = np.zeros_like(snrs) if failure_rates is None else np.asarray(failure_rates, float)
    order = np.argsort(snrs)
    snrs, deltas, fr = snrs[order], deltas[order], fr[order]
    ok = (deltas <= tol_db) & (fr <= max_failure)
    for i in range(len(snrs)):
        if ok[i:].all():
            return float(snrs[i])
    return float("nan")


def empirical_cdf(values):
    v = np.sort(np.asarray(values, float))
    return v, np.arange(1, v.size + 1) / v.size


def match_paths_by_angle(est_aod, est_aoa, true_aod, true_aoa):
    """Greedy one-to-one matching of estimated to true paths on (sin AoD, sin AoA). Returns (est_idx, true_idx)."""
    from scipy.optimize import linear_sum_assignment
    ea = np.stack([np.sin(est_aod), np.sin(est_aoa)], 1)
    ta = np.stack([np.sin(true_aod), np.sin(true_aoa)], 1)
    cost = np.linalg.norm(ea[:, None, :] - ta[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return r, c

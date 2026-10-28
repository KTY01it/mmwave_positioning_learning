"""P2 evaluation harness shared by every method (classical, AI, hybrid). Frozen protocol: docs/evaluation_protocol.md.

One record per (method, sample, G). Bounds are computed for the SAME beam subset the method sees. Errors are
normalised per sample (P2.2 way 2); failures are kept (NaN error, counted as outliers).
"""
from __future__ import annotations

import time

import numpy as np
import yaml

from ..bounds.peb_reb import compute_bounds
from ..config import CONFIG_DIR
from ..models.base import Estimator
from ..system.simulator import SimulationOutput, subsample_beams
from . import metrics as M

G_GRID = (4, 8, 12, 16, 20, 24, 32)
# catastrophic-failure distance, fixed on 2026-10-06 before any step-6B test (docs/evaluation_protocol.md)
CATASTROPHIC_DISTANCE_M = 1.0
SNR_BIN_EDGES = tuple(np.arange(-47.5, 25.0, 5.0))  # bins centred on multiples of 5 dB


def with_estimator_profile(sample: SimulationOutput, profile: str) -> SimulationOutput:
    """Same observation, estimator settings from configs/estimators/<profile>.yaml (experiment keys kept)."""
    import dataclasses
    prof = yaml.safe_load((CONFIG_DIR / "estimators" / f"{profile}.yaml").read_text())["estimator"]
    cfg = sample.context.config.replace(**{f"estimator.{k}": v for k, v in prof.items()
                                          if k not in ("olos_alpha_step",)})
    return dataclasses.replace(sample, context=dataclasses.replace(sample.context, config=cfg))


def _channel_errors(est, gt, crb):
    """Mean over true paths of (error / sqrt(CRB))^2 for tau, AoD, AoA; NaN where a path is missed."""
    out = {"chan_eps_tau": np.nan, "chan_eps_aod": np.nan, "chan_eps_aoa": np.nan, "paths_detected": 0.0}
    if est is None or est.num_paths == 0 or crb is None:
        return out
    r, c = M.match_paths_by_angle(est.aod, est.aoa, gt.paths.aod, gt.paths.aoa)
    if not len(c):
        return out
    out["chan_eps_tau"] = float(np.mean(((est.delay[r] - gt.paths.delay[c]) / crb.crb_delay[c]) ** 2))
    out["chan_eps_aod"] = float(np.mean((M.wrap_angle(est.aod[r] - gt.paths.aod[c]) / crb.crb_aod[c]) ** 2))
    out["chan_eps_aoa"] = float(np.mean((M.wrap_angle(est.aoa[r] - gt.paths.aoa[c]) / crb.crb_aoa[c]) ** 2))
    out["paths_detected"] = float(len(c) / gt.num_paths)
    return out


def evaluate_sample(estimator: Estimator, sample: SimulationOutput, G_list=None, method: str | None = None,
                    tags: dict | None = None) -> list[dict]:
    """Run ``estimator`` on ``sample`` restricted to each G in G_list (default: the full beam set)."""
    recs = []
    for G in (G_list or [sample.y.shape[1]]):
        s = subsample_beams(sample, G) if G != sample.y.shape[1] else sample
        try:
            b = compute_bounds(s)
        except np.linalg.LinAlgError:
            b = None
        t0 = time.perf_counter()
        try:
            out = estimator(s)
            err = None
        except Exception as exc:  # a method failing on a sample is a result, not a crash
            out, err = None, repr(exc)
        wall = time.perf_counter() - t0
        pos = float(M.position_error(out.position, s.gt.position)) if out is not None else np.nan
        ori = float(M.angle_error(out.orientation, s.gt.orientation)) if out is not None and np.isfinite(out.orientation) else np.nan
        peb = b.peb if b is not None else np.nan
        reb = b.reb if b is not None else np.nan
        timing = (out.info.get("timing", {}) if out is not None else {})
        ch = out.channel if out is not None else None
        rec = {
            "method": method or getattr(estimator, "name", type(estimator).__name__),
            "oracle": bool(getattr(estimator, "requires_oracle", False)),
            "seed": s.meta.seed, "condition": s.gt.condition, "G": int(G), "snr": float(s.meta.snr_db),
            "K": s.gt.num_paths, "K_hat": None if ch is None else int(ch.num_paths),
            "pos_err": pos, "ori_err": ori, "peb": peb, "reb": reb,
            "eps_pos": pos ** 2 / peb ** 2 if np.isfinite(peb) else np.nan,
            "eps_ori": ori ** 2 / reb ** 2 if np.isfinite(reb) else np.nan,
            "failed": not np.isfinite(pos), "error": err,
            "t_total": wall, "t_coarse": timing.get("coarse", np.nan), "t_sage": timing.get("sage", np.nan),
            "t_localize": timing.get("localize", np.nan),
            "sage_iterations": int(out.info["sage"].iterations) if out is not None and "sage" in out.info else None,
            **_channel_errors(ch, s.gt, b),
            **(tags or {}),
        }
        recs.append(rec)
    return recs


def snr_bin(snr: float) -> float:
    return float(5.0 * np.round(snr / 5.0))


def summarize(records: list[dict], by=("method", "condition", "G", "snr_bin")) -> list[dict]:
    """Aggregate records into the standard table (one row per group)."""
    groups: dict = {}
    for r in records:
        r = {**r, "snr_bin": snr_bin(r["snr"])}
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    rows = []
    for key, R in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        pos = np.array([r["pos_err"] for r in R], float)
        ori = np.array([r["ori_err"] for r in R], float)
        peb = np.array([r["peb"] for r in R], float)
        eps = np.array([r["eps_pos"] for r in R], float)
        epa = np.array([r["eps_ori"] for r in R], float)
        row = dict(zip(by, key))
        row.update({
            "n": len(R),
            "rmse_pos": M.rmse(pos), "median_pos": float(np.nanmedian(pos)) if np.isfinite(pos).any() else np.nan,
            "p90_pos": float(np.nanpercentile(pos, 90)) if np.isfinite(pos).any() else np.nan,
            "rmse_ori": M.rmse(ori),
            "eps_pos_db": float(10 * np.log10(np.nanmean(eps))) if np.isfinite(eps).any() else np.nan,
            "median_eps_pos_db": float(10 * np.log10(np.nanmedian(eps))) if np.isfinite(eps).any() else np.nan,
            "eps_ori_db": float(10 * np.log10(np.nanmean(epa))) if np.isfinite(epa).any() else np.nan,
            "p_out": M.outlier_rate(pos, peb),
            "p_cat": float(np.mean(~(pos <= CATASTROPHIC_DISTANCE_M))),  # NaN (failure) counts as catastrophic
            "failure": float(np.mean([r["failed"] for r in R])),
            "k_correct": float(np.mean([r["K_hat"] == r["K"] for r in R])),
            "chan_eps_tau_db": float(10 * np.log10(np.nanmean([r["chan_eps_tau"] for r in R])))
            if np.isfinite([r["chan_eps_tau"] for r in R]).any() else np.nan,
            "t_total": float(np.mean([r["t_total"] for r in R])),
            "t_coarse": float(np.nanmean([r["t_coarse"] for r in R])) if np.isfinite([r["t_coarse"] for r in R]).any() else np.nan,
            "t_sage": float(np.nanmean([r["t_sage"] for r in R])) if np.isfinite([r["t_sage"] for r in R]).any() else np.nan,
            "t_localize": float(np.nanmean([r["t_localize"] for r in R])) if np.isfinite([r["t_localize"] for r in R]).any() else np.nan,
        })
        rows.append(row)
    return rows

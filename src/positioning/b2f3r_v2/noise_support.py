"""T3 (v2): noise level and stopping without the simulator's N0, with protection against weak-path self-masking.

Cells: unitary (delay, beam, receive-beam) transform of the pilot-derotated residual (positioning.robust.noise). Under
the null hypothesis every cell power is exponential with mean N0, and a "column" (one delay x receive-beam cell over
the G beams) has energy N0 * Gamma(G, 1). A path occupies essentially one column plus leakage into its neighbours.

  N1  "resid"     median of all cell powers / ln 2; stop ||r||^2 / N0_hat < M (1 + A99 / sqrt M)        (B2f3R-v1)
  N2  "censored"  noise-support alternation: columns whose energy exceeds a fixed null quantile (and their +-1
                  delay / receive-beam neighbours) are treated as unexplained structure and excluded; N0_hat is the
                  median of the remaining cells / ln 2, divided by a bias factor; repeated until the excluded set no
                  longer changes (at most CENSOR_ITERS times). Stop as N1 with its own null quantile.
  N3  "cfar"      N2 plus a second, scale-invariant test: the largest column energy / N0_hat against its null quantile
                  (order-statistic CFAR principle). The model order grows while either test rejects. Each test at 0.01,
                  so the union is not at the frozen 0.01 (construction error, recorded).
  N3b "cfar_b"    N3 with both tests at their 0.995 null quantiles (Bonferroni), union at about 0.01.

All constants are Monte Carlo properties of the noise-only null distribution of the actual cell transform (G, N, Nr of
the system); they are computed by `calibrate()` without any development sample and stored in CAL. [DESIGN]
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.stats import gamma as gamma_dist

from ..b2f3r import noise as NZ1
from ..robust.noise import cell_powers

LN2 = float(np.log(2.0))
CENSOR_P = 1e-3          # per-column null exceedance probability that marks a column as structure
CENSOR_ITERS = 4
CAL_FILE = Path(__file__).with_name("null_calibration.json")


def _cols(p, shape):
    """cell powers (flat) -> column energies [delay, rx-beam] (sum over the G beams)."""
    return p.reshape(shape).sum(axis=1)


def censored_floor(r, context, kappa: float = 1.0, return_info: bool = False):
    """Noise-support alternation (N2). kappa: null bias factor of the censored median (from CAL)."""
    N, G, Nr = r.shape
    p = cell_powers(r, context)
    shape = (N, G, Nr)
    E = _cols(p, shape)
    thr = gamma_dist.ppf(1 - CENSOR_P, G)
    n0 = float(np.median(p) / LN2)
    mask = np.zeros_like(E, bool)
    it = 0
    for it in range(1, CENSOR_ITERS + 1):
        flag = E > thr * n0
        new = flag | np.roll(flag, 1, 0) | np.roll(flag, -1, 0) | np.roll(flag, 1, 1) | np.roll(flag, -1, 1)
        keep = np.broadcast_to(~new[:, None, :], shape).ravel()
        n0_new = float(np.median(p[keep]) / LN2 / kappa) if keep.any() else n0
        done = np.array_equal(new, mask)
        mask, n0 = new, n0_new
        if done:
            break
    if return_info:
        return n0, {"iters": it, "censored_cols": int(mask.sum()), "E": E}
    return n0


@functools.lru_cache(maxsize=1)
def _cal():
    return json.loads(CAL_FILE.read_text()) if CAL_FILE.exists() else {}


def constants(G: int) -> dict:
    c = _cal().get(str(G))
    if c is None:
        raise KeyError(f"no null calibration for G = {G}; run positioning.b2f3r_v2.noise_support.calibrate")
    return c


def level(kind: str, r, context) -> float:
    if kind == "resid":
        return NZ1.residual_floor(r, context)
    return censored_floor(r, context, constants(r.shape[1])["kappa"])


def stop_ok(kind: str, f: float, n0: float, r, context) -> bool:
    """True if the residual is consistent with noise (the forward loop does not run)."""
    M = r.size
    if kind == "resid":
        return f / n0 < NZ1.ratio_threshold(M)
    c = constants(r.shape[1])
    a, q = (c["a995"], c["max_q995"]) if kind == "cfar_b" else (c["a99"], c["max_q99"])
    if not f / n0 < M * (1.0 + a / np.sqrt(M)):
        return False
    if kind in ("cfar", "cfar_b"):
        E = _cols(cell_powers(r, context), r.shape)
        return float(E.max()) / n0 < q
    return True


def calibrate(context, G_list=(6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32), draws: int = 4000, seed: int = 20261015):
    """Noise-only Monte Carlo of the censored estimator and of both stop statistics for the system of `context`
    (pilots must be unit-modulus; the cell transform of white noise is white). Pass 1 fixes the bias factor kappa,
    pass 2 (fresh draws) the 0.99 quantiles of ||r||^2 / (M N0_hat) and max column energy / N0_hat. Writes CAL_FILE."""
    import dataclasses
    N, Nr = context.config.physics.num_subcarriers, context.config.array.num_rx
    out = {}
    for G in G_list:
        x = np.tile(context.x, (1, int(np.ceil(G / context.x.shape[1])), 1))[:, :G]
        ctx = dataclasses.replace(context, x=x)
        M = N * G * Nr
        noise = lambda rng: (rng.standard_normal((N, G, Nr)) + 1j * rng.standard_normal((N, G, Nr))) * np.sqrt(0.5)
        rng = np.random.default_rng([seed, G, 1])
        kappa = float(np.median([censored_floor(noise(rng), ctx, 1.0) for _ in range(draws)]))
        rng = np.random.default_rng([seed, G, 2])
        ratio, mx, its, cens = [], [], [], []
        for _ in range(draws):
            w = noise(rng)
            n0, info = censored_floor(w, ctx, kappa, return_info=True)
            ratio.append(float(np.sum(np.abs(w) ** 2) / n0 / M))
            mx.append(float(info["E"].max() / n0))
            its.append(info["iters"]); cens.append(info["censored_cols"])
        ratio, mx = np.array(ratio), np.array(mx)
        out[str(G)] = {"kappa": kappa, "a99": float((np.quantile(ratio, 0.99) - 1) * np.sqrt(M)),
                       "max_q99": float(np.quantile(mx, 0.99)), "ratio_sd_sqrtM": float(ratio.std() * np.sqrt(M)),
                       "a995": float((np.quantile(ratio, 0.995) - 1) * np.sqrt(M)), "max_q995": float(np.quantile(mx, 0.995)),
                       "median_ratio": float(np.median(ratio)), "mean_iters": float(np.mean(its)),
                       "mean_censored_cols": float(np.mean(cens)), "draws": draws, "seed": seed}
        print(G, out[str(G)], flush=True)
    CAL_FILE.write_text(json.dumps(out, indent=1))
    _cal.cache_clear()
    return out

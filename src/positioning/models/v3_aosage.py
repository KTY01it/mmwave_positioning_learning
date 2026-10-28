"""V3-1: adaptive model-order SAGE (E. T. R. Pinto, M. Juntti, "Robust OFDM-SAGE Channel Estimation Algorithm With
Adaptive Model Order", IEEE TWC 25:5275-5290, 2026, DOI 10.1109/TWC.2025.3617244), adapted to the Shahmansoori
single-BS MIMO-OFDM model and initialised by the frozen learned path heatmap (v1R).

[PAPER]  log L = -||Y - Y(xi)||^2 / N0 (8). Paths are added one at a time, each only after the current model has
         converged. A new path is kept iff (f_{L-1} - f_L) / N0 > z_N(delta) (29), and failed paths are re-drawn.
         Global stop: f_L / N0 < eps = 0.5 * chi2inv(1 - P_FA, 2 * #complex observations) (30)-(31). No removal.
         delta = 1e-8, P_FA = 0.01.
[ADAPTATION] z_N(delta) re-derived for 3 parameters per path (tau, AoD, AoA) with the paper's Euler-characteristic
         method (scratchpad derivation): z = 66.7 at delta = 1e-8 (the paper's closed form plugged with our sizes
         gives about 80 because its Lambda assumes 4 parameters and an isotropic pilot).
         Initial model = the network's confident paths (calibrated threshold) instead of L = 1 from random draws.
         Re-draw sources: the remaining network candidates in decreasing existence probability, then the peak of the
         residual angle-delay map (a matched-filter full-range search). These replace uniform random draws, because
         our SAGE line search is local and the paper's exact global 1-D updates do not apply to frequency-dependent
         arrays / random beams. Every SAGE iteration updates all paths (paper: newest path + window).
[DESIGN]  L_max = 6, at most R_MAX re-draws per model order, SAGE capped at IT_MID iterations per middle loop.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import chi2

from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage
from .path_init import PathInitEstimator, decode, ls_gains, network_input

Z_DELTA = {1e-8: 66.7, 1e-4: 47.6, 1e-2: 38.0}
L_MAX, R_MAX, IT_MID = 6, 4, 10


def _fit(model, y, est):
    r = y - model.mean(est)
    return float(np.real(np.vdot(r, r)))


def _near(c, est, Ts, nt, nr):
    return any(abs(c[3] - est.delay[k]) <= Ts and abs(np.sin(c[1]) - np.sin(est.aod[k])) <= 2.0 / nt
               and abs(np.sin(c[2]) - np.sin(est.aoa[k])) <= 2.0 / nr for k in range(est.num_paths))


def aosage(pe: PathInitEstimator, y, context, delta: float = 1e-8, pfa: float = 0.01, global_stop: bool = True,
           x=None, lobe_fix: bool = True, dup_reject: bool = True, backward: bool = False):
    """lobe_fix [DESIGN, V2-A]: AoD of every initial path = maximiser of the single-path statistic on the residual of
    the other initial paths. dup_reject [DESIGN]: a new path that converges into the resolution cell of an existing path
    is not a new component (it absorbs residual fit error of the existing one) and is rejected before the GLRT."""
    from .v2_paths import _aod_scan, _peak, _residual_candidate, OS
    cfg = context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    model = ObservationModel(context)
    t0 = time.perf_counter()
    x = network_input(y, context) if x is None else x
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    cand = [(p, aod, aoa, tau * Ts) for p, aod, aoa, tau in decode(out, cfg, 0.05, 8)]
    conf = [c for c in cand if c[0] >= pe.threshold] or cand[:1]
    pool = [c for c in cand if c[0] < pe.threshold]
    grid = np.arcsin(np.clip(np.linspace(-1, 1, OS * nt, endpoint=False) + 1.0 / (OS * nt), -1, 1))
    tx_proj = model.tx_projection(grid)
    if lobe_fix:
        fixed = []
        for i, c in enumerate(conf):
            others = conf[:i] + conf[i + 1:]
            r = y
            if others:
                do, ao, bo = (np.array([q[j] for q in others]) for j in (3, 1, 2))
                g = ls_gains(model, y, do, ao, bo)
                r = y - model.mean(ChannelEstimate(do, ao, bo, g))
            fixed.append((c[0], _peak(grid, _aod_scan(model, r, c[3], c[2], grid, tx_proj)), c[2], c[3]))
        conf = fixed
    d, a, b = (np.array([c[i] for c in conf]) for i in (3, 1, 2))
    est = sage(y, context, ChannelEstimate(d, a, b, ls_gains(model, y, d, a, b)), max_iter=IT_MID,
               tol=cfg.estimator.sage_tol).estimate
    f = _fit(model, y, est)
    z = Z_DELTA[delta]
    eps = 0.5 * chi2.ppf(1 - pfa, 2 * y.size)
    trials = accepted = 0
    stop = "lmax"
    while est.num_paths < L_MAX:
        if global_stop and f / context.n0 < eps:
            stop = "global"
            break
        added = False
        for _ in range(R_MAX):                                    # re-draws for this model order
            pool = [c for c in pool if not _near(c, est, Ts, nt, nr)]
            if pool:
                c = pool.pop(0)
            else:
                r = y - model.mean(est)
                c = _residual_candidate(model, r, context,
                                        [(0, est.aod[k], est.aoa[k], est.delay[k]) for k in range(est.num_paths)],
                                        grid, tx_proj)
            trials += 1
            g0 = ls_gains(model, y - model.mean(est), np.array([c[3]]), np.array([c[1]]), np.array([c[2]]))
            trial = ChannelEstimate(np.r_[est.delay, c[3]], np.r_[est.aod, c[1]], np.r_[est.aoa, c[2]],
                                    np.r_[est.gain_tilde, g0])
            trial = sage(y, context, trial, max_iter=IT_MID, tol=cfg.estimator.sage_tol).estimate
            if dup_reject and _near((0, trial.aod[-1], trial.aoa[-1], trial.delay[-1]), trial.subset(list(range(trial.num_paths - 1))), Ts, nt, nr):
                continue
            f_new = _fit(model, y, trial)
            if (f - f_new) / context.n0 > z:                      # GLRT (29)
                est, f, added = trial, f_new, True
                accepted += 1
                break
        if not added:
            stop = "redraws"
            break
    removed = 0
    if backward:                     # [DESIGN] backward GLRT with the same z: drop paths whose removal costs < z
        while est.num_paths > 1:
            costs = []
            for k in range(est.num_paths):
                sub = est.subset([j for j in range(est.num_paths) if j != k])
                g = ls_gains(model, y, sub.delay, sub.aod, sub.aoa)
                costs.append(_fit(model, y, ChannelEstimate(sub.delay, sub.aod, sub.aoa, g)))
            f_all = _fit(model, y, ChannelEstimate(est.delay, est.aod, est.aoa,
                                                   ls_gains(model, y, est.delay, est.aod, est.aoa)))
            k = int(np.argmin(costs))
            if (costs[k] - f_all) / context.n0 > z:
                break
            est = est.subset([j for j in range(est.num_paths) if j != k])
            removed += 1
        if removed:
            est = sage(y, context, ChannelEstimate(est.delay, est.aod, est.aoa,
                                                   ls_gains(model, y, est.delay, est.aod, est.aoa)),
                       max_iter=2, tol=cfg.estimator.sage_tol).estimate
            f = _fit(model, y, est)
    est.info.update({"aosage": {"trials": trials, "accepted": accepted, "removed": removed, "stop": stop, "f_over_n0": f / context.n0,
                                "eps": eps, "z": z, "t": time.perf_counter() - t0}})
    return est


def aosage2(pe: PathInitEstimator, y, context, it_mid: int = IT_MID, kappa: float = 1.0, stepwise: bool = False,
            oracle_k: int | None = None, delta: float = 1e-8, pfa: float = 0.01, x=None):
    """V3-2A variants of the V3-1 estimator (aosage with lobe_fix=False, dup_reject=True, backward=True).

    it_mid    SAGE iterations per middle loop (V3-1: 10; "converged": 40 with the strong tolerance).
    kappa     GLRT / global-stop variance sigma2_res = kappa * N0. N0 stays the physical noise variance; kappa >= 1 is a
              separately calibrated nuisance factor for residual model-fit error (calib_v2, true-K converged fits).
    stepwise  [DESIGN] alternate forward and backward GLRT passes until the model order is stable (max 3 rounds).
    oracle_k  DIAGNOSTIC ONLY: the true K is given; forward adds the best of the re-draws until K paths, backward
              removes the weakest paths down to K; all parameters are still estimated.
    """
    from .v2_paths import _residual_candidate, OS
    cfg = context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    model = ObservationModel(context)
    tol = cfg.estimator.sage_tol
    t0 = time.perf_counter()
    x = network_input(y, context) if x is None else x
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    cand = [(p, aod, aoa, tau * Ts) for p, aod, aoa, tau in decode(out, cfg, 0.05, 8)]
    conf = [c for c in cand if c[0] >= pe.threshold] or cand[:1]
    pool = [c for c in cand if c[0] < pe.threshold]
    grid = np.arcsin(np.clip(np.linspace(-1, 1, OS * nt, endpoint=False) + 1.0 / (OS * nt), -1, 1))
    tx_proj = model.tx_projection(grid)
    d, a, b = (np.array([c[i] for c in conf]) for i in (3, 1, 2))
    est = sage(y, context, ChannelEstimate(d, a, b, ls_gains(model, y, d, a, b)), max_iter=it_mid, tol=tol).estimate
    f = _fit(model, y, est)
    s2 = kappa * context.n0
    z = Z_DELTA[delta]
    eps = 0.5 * chi2.ppf(1 - pfa, 2 * y.size)
    n_sage = 1
    stats = {"accepted": 0, "removed": 0, "rounds": 0}

    def forward(est, f, pool):
        nonlocal n_sage
        changed = False
        while est.num_paths < (oracle_k if oracle_k else L_MAX):
            if oracle_k is None and f / s2 < eps:
                break
            best = None
            for _ in range(R_MAX):
                pool = [c for c in pool if not _near(c, est, Ts, nt, nr)]
                if pool:
                    c = pool.pop(0)
                else:
                    c = _residual_candidate(model, y - model.mean(est), context,
                                            [(0, est.aod[k], est.aoa[k], est.delay[k]) for k in range(est.num_paths)],
                                            grid, tx_proj)
                g0 = ls_gains(model, y - model.mean(est), np.array([c[3]]), np.array([c[1]]), np.array([c[2]]))
                trial = sage(y, context, ChannelEstimate(np.r_[est.delay, c[3]], np.r_[est.aod, c[1]],
                                                         np.r_[est.aoa, c[2]], np.r_[est.gain_tilde, g0]),
                             max_iter=it_mid, tol=tol).estimate
                n_sage += 1
                if _near((0, trial.aod[-1], trial.aoa[-1], trial.delay[-1]),
                         trial.subset(list(range(trial.num_paths - 1))), Ts, nt, nr):
                    continue
                f_new = _fit(model, y, trial)
                if oracle_k:
                    if best is None or f_new < best[1]:
                        best = (trial, f_new)
                    continue
                if (f - f_new) / s2 > z:
                    best = (trial, f_new)
                    break
            if best is None:
                break
            est, f = best
            stats["accepted"] += 1
            changed = True
        return est, f, pool, changed

    def backward(est, f):
        nonlocal n_sage
        changed = False
        while est.num_paths > (oracle_k if oracle_k else 1):
            costs = []
            for k in range(est.num_paths):
                sub = est.subset([j for j in range(est.num_paths) if j != k])
                costs.append(_fit(model, y, ChannelEstimate(sub.delay, sub.aod, sub.aoa,
                                                            ls_gains(model, y, sub.delay, sub.aod, sub.aoa))))
            f_all = _fit(model, y, ChannelEstimate(est.delay, est.aod, est.aoa,
                                                   ls_gains(model, y, est.delay, est.aod, est.aoa)))
            k = int(np.argmin(costs))
            if oracle_k is None and (costs[k] - f_all) / s2 > z:
                break
            est = est.subset([j for j in range(est.num_paths) if j != k])
            stats["removed"] += 1
            changed = True
        if changed:
            est = sage(y, context, ChannelEstimate(est.delay, est.aod, est.aoa,
                                                   ls_gains(model, y, est.delay, est.aod, est.aoa)),
                       max_iter=it_mid, tol=tol).estimate
            n_sage += 1
            f = _fit(model, y, est)
        return est, f, changed

    for rnd in range(3 if stepwise else 1):
        stats["rounds"] = rnd + 1
        est, f, pool, ch1 = forward(est, f, pool)
        est, f, ch2 = backward(est, f)
        if not (ch1 or ch2):
            break
    est.info.update({"aosage": {**stats, "n_sage": n_sage, "f_over_n0": f / context.n0, "kappa": kappa,
                                "t": time.perf_counter() - t0}})
    return est

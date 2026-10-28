"""Model-order loop of the B2f3-Robust candidates: control flow of positioning.models.v3_local.aosage3 (variant "B2f")
with the frozen helper functions and thresholds, plus three optional, observation-only extensions.

  noise = "known"   context.n0 is used as given by the caller (exact, or a plug-in estimate); frozen chi-square stop
  noise = "resid"   noise level re-estimated from the residual of the current model; unknown-scale stop (b2f3r.noise)
  switch = True     selective support repair (T1, A2): if the learned initial support fails the global stop right after
                    its first fit, the classical 4x coarse stage supplies the initial support instead
  cand_fn           replaces the initial candidate set unconditionally (A1 = robust.lowg.cf_hybrid)

With noise = "known", switch = False and cand_fn = None the loop reproduces V3Candidate exactly (tests/test_b2f3r.py).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
from scipy.stats import chi2

from ..models import v3_local as VL
from ..models.path_init import decode, ls_gains
from ..models.v3_aosage import L_MAX, R_MAX, Z_DELTA, _fit
from ..robust import loop as LP
from ..robust import lowg as LG
from ..signal_processing import delay_estimation
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage
from . import noise as NZ

Z = Z_DELTA[1e-8]
PFA = 0.01


def model_order_loop(pe, y, context, x, *, it_mid=3, noise="known", switch=False, cand_fn=None, log=None):
    """Returns (est, f, n0): final channel estimate, residual energy, and the noise level in force at the end."""
    log = {} if log is None else log
    log.update({"n_sage": 0, "sweeps": 0, "rounds": 0, "fwd_entered": 0, "accepted": 0, "removed": 0, "fwd_tests": 0,
                "switched": 0})
    cfg = context.config
    Ts = cfg.physics.sampling_period
    model = ObservationModel(context)
    tol = cfg.estimator.sage_tol
    M = y.size
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    cand = [(p, aod, aoa, tau * Ts) for p, aod, aoa, tau in decode(out, cfg, 0.05, 8)]
    conf = [c for c in cand if c[0] >= pe.threshold] or cand[:1]
    pool = [c for c in cand if c[0] < pe.threshold]
    if cand_fn is not None:
        conf, pool = cand_fn(conf, pool, y, context, model, out)
    eps = 0.5 * chi2.ppf(1 - PFA, 2 * M) if noise == "known" else NZ.ratio_threshold(M)

    def level(est):
        return context.n0 if noise == "known" else NZ.residual_floor(y - model.mean(est), context)

    def refit(e):
        r = sage(y, context, e, max_iter=it_mid, tol=tol)
        log["n_sage"] += 1
        log["sweeps"] += int(r.iterations)
        return VL.fold_branches(r.estimate, cfg)

    def start(c):
        d, a, b = (np.array([q[i] for q in c]) for i in (3, 1, 2))
        e = refit(ChannelEstimate(d, a, b, ls_gains(model, y, d, a, b)))
        return e, _fit(model, y, e)

    est, f = start(conf)
    n0 = level(est)
    if switch and not f / n0 < eps:
        conf2, pool2 = LG.cf_hybrid(conf, pool, y, dataclasses.replace(context, n0=n0), model, out)
        if conf2 is not conf:
            pool = pool2
            est, f = start(conf2)
            n0 = level(est)
            log["switched"] = 1

    def proposals(est):
        r = y - model.mean(est)
        centers = [(est.aod[k], est.aoa[k]) for k in range(est.num_paths)] + [VL._residual_peak(model, r, context)]
        loc, _ = VL.local_candidates(model, context, r, centers)
        props = [("local", q[1], q[2], q[3], q[4]) for q in loc]
        for c in pool:
            props.append(("pool", c[3], c[1], c[2], None))
        rp = VL._residual_peak(model, r, context)
        props.append(("peak", None, rp[0], rp[1], None))
        return props

    def forward(est, f, n0):
        changed = False
        while est.num_paths < L_MAX:
            if f / n0 < eps:
                break
            log["fwd_entered"] += 1
            best = None
            r = y - model.mean(est)
            for src, tau, aod, aoa, g in proposals(est)[:R_MAX + est.num_paths + 1]:
                if tau is None:
                    bb = (model.tx_projection(aod)[:, :, None] * model.rx_response(aoa)[:, None, :]).reshape(model.N, -1)
                    rb = r.reshape(model.N, -1)
                    coef = np.einsum("nm,nm->n", bb.conj(), rb) / np.maximum(np.sum(np.abs(bb) ** 2, axis=1), 1e-30)
                    tau = delay_estimation.periodogram_delay(coef, Ts, cfg.estimator.delay_oversampling)
                if g is None:
                    g = ls_gains(model, r, np.array([tau]), np.array([aod]), np.array([aoa]))[0]
                trial = refit(ChannelEstimate(np.r_[est.delay, tau], np.r_[est.aod, aod], np.r_[est.aoa, aoa],
                                              np.r_[est.gain_tilde, g]))
                f_new = _fit(model, y, trial)
                log["fwd_tests"] += 1
                if (f - f_new) / n0 > Z:
                    best = (trial, f_new)
                    break
            if best is None:
                break
            est, f = best
            n0 = level(est)
            log["accepted"] += 1
            changed = True
        return est, f, n0, changed

    def backward(est, f, n0):
        est, changed = LP.backward_prune(model, y, est, n0, log)
        if changed:
            est = refit(ChannelEstimate(est.delay, est.aod, est.aoa, ls_gains(model, y, est.delay, est.aod, est.aoa)))
            f = _fit(model, y, est)
            n0 = level(est)
        return est, f, n0, changed

    for _ in range(3):
        log["rounds"] += 1
        est, f, n0, c1 = forward(est, f, n0)
        est, f, n0, c2 = backward(est, f, n0)
        if not (c1 or c2):
            break
    log["stop_ok"] = int(f / n0 < eps)
    return est, f, float(n0)

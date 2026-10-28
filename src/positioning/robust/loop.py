"""Candidate copy of the B2f3 model-order loop (control flow of positioning.models.v3_local.aosage3, variant "B2f"),
calling the frozen helper functions. The frozen file is not edited; tests/test_robust.py asserts that, with the default
arguments, this copy reproduces V3Candidate exactly.

Extensions used by the program (all optional):
  cand_fn(conf, pool, y, context, model, netout) -> (conf, pool)   a different initial candidate set (R2)
  stop = "F"   exact variance-ratio test of the residual against nu independent noise-only samples (R1 diagnostic)
  post_fn(est, f, y, context, model, refit) -> (est, f)            a step after the loop (R2, variant L2)
The frozen thresholds z = Z_DELTA[1e-8] and P_FA = 0.01 are used unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2, f as fdist

from ..models import v3_local as VL
from ..models.path_init import decode, ls_gains, network_input
from ..models.router import branch_localize, classical_route
from ..models.v3_aosage import L_MAX, R_MAX, Z_DELTA, _fit
from ..signal_processing import delay_estimation
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage

Z = Z_DELTA[1e-8]
PFA = 0.01


def backward_prune(model, y, est: ChannelEstimate, n0: float, log=None):
    """The frozen backward path test (v3_local.aosage3::backward) without the refit. Returns (est, changed)."""
    changed = False
    while est.num_paths > 1:
        costs = []
        for k in range(est.num_paths):
            sub = est.subset([j for j in range(est.num_paths) if j != k])
            costs.append(_fit(model, y, ChannelEstimate(sub.delay, sub.aod, sub.aoa,
                                                        ls_gains(model, y, sub.delay, sub.aod, sub.aoa))))
        f_all = _fit(model, y, ChannelEstimate(est.delay, est.aod, est.aoa,
                                               ls_gains(model, y, est.delay, est.aod, est.aoa)))
        k = int(np.argmin(costs))
        if (costs[k] - f_all) / n0 > Z:
            break
        est = est.subset([j for j in range(est.num_paths) if j != k])
        changed = True
        if log is not None:
            log["removed"] += 1
    return est, changed


def model_order_loop(pe, y, context, x, *, it_mid=3, cand_fn=None, stop="chi2", n0_ind=None, nu=None, post_fn=None,
                     log=None):
    log = {} if log is None else log
    log.update({"n_sage": 0, "sweeps": 0, "rounds": 0, "fwd_entered": 0, "accepted": 0, "removed": 0, "fwd_tests": 0})
    cfg = context.config
    Ts = cfg.physics.sampling_period
    model = ObservationModel(context)
    tol = cfg.estimator.sage_tol
    n0 = context.n0
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    cand = [(p, aod, aoa, tau * Ts) for p, aod, aoa, tau in decode(out, cfg, 0.05, 8)]
    conf = [c for c in cand if c[0] >= pe.threshold] or cand[:1]
    pool = [c for c in cand if c[0] < pe.threshold]
    if cand_fn is not None:
        conf, pool = cand_fn(conf, pool, y, context, model, out)
    M = y.size
    eps = 0.5 * chi2.ppf(1 - PFA, 2 * M)

    def refit(e):
        r = sage(y, context, e, max_iter=it_mid, tol=tol)
        log["n_sage"] += 1
        log["sweeps"] += int(r.iterations)
        return VL.fold_branches(r.estimate, cfg)

    d, a, b = (np.array([c[i] for c in conf]) for i in (3, 1, 2))
    est = refit(ChannelEstimate(d, a, b, ls_gains(model, y, d, a, b)))
    f = _fit(model, y, est)

    def stop_now(f, k):
        if stop == "chi2":
            return f / n0 < eps
        dfn = 2 * (M - 2.5 * k)                                        # F test against nu noise-only samples
        return (f / (M - 2.5 * k)) / n0_ind < fdist.ppf(1 - PFA, dfn, 2 * nu)

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

    def forward(est, f):
        changed = False
        while est.num_paths < L_MAX:
            if stop_now(f, est.num_paths):
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
            log["accepted"] += 1
            changed = True
        return est, f, changed

    def backward(est, f):
        est, changed = backward_prune(model, y, est, n0, log)
        if changed:
            est = refit(ChannelEstimate(est.delay, est.aod, est.aoa, ls_gains(model, y, est.delay, est.aod, est.aoa)))
            f = _fit(model, y, est)
        return est, f, changed

    for _ in range(3):
        log["rounds"] += 1
        est, f, c1 = forward(est, f)
        est, f, c2 = backward(est, f)
        if not (c1 or c2):
            break
    if post_fn is not None:
        est, f = post_fn(est, f, y, context, model, refit)
    return est, f


def run(candidate, y, context, **kw):
    """End to end with the copied loop and the frozen condition test and localisation.
    `candidate` is a frozen V3Candidate (its networks and it_mid are used); context.n0 is the noise input."""
    log = {}
    x = network_input(y, context)
    est, f = model_order_loop(candidate.pe, y, context, x, it_mid=candidate.it_mid, log=log, **kw)
    cond, _ = classical_route(est, context)
    out = branch_localize(est, y, context, cond, candidate.geom)
    out.info.update({"c_hat": cond, "loop": log})
    return out

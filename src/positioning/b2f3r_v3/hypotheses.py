"""V3-A: a bounded set of path-support hypotheses, each refined with the frozen signal model (SAGE inside the frozen
model-order loop of B2f3R-v2; RELAX-style cyclic re-estimation for H2). No ground truth is used.

  H0  B2f3R-v2 path stage (network support; A2b at G <= 10)                        [v2, unchanged]
  H1  classical 4x coarse support (DCS-SOMP), then the same loop                     [v2 G2 path stage]
  H2  network support after cyclic global AoD search with successive cancellation    [RELAX principle; R2 candidate L1]
  Hp  H0 plus the best single additional path proposal, accepted regardless of the global stop, SAGE refit  [DESIGN]
  Hm  H0 minus its least significant path (smallest residual increase on removal), SAGE refit           [DESIGN]
Duplicates (same number of paths, every path within one delay/AoA/AoD cell of the other hypothesis) are detected by
`same_support` so that identical supports are not localised twice.
"""
from __future__ import annotations

import time

import numpy as np

from ..models import v3_local as VL
from ..models.path_init import ls_gains
from ..models.v3_aosage import _fit
from ..robust import lowg as LG
from ..signal_processing import delay_estimation
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage


def _refit(y, context, e, it_mid=3):
    r = sage(y, context, e, max_iter=it_mid, tol=context.config.estimator.sage_tol)
    return VL.fold_branches(r.estimate, context.config)


def plus_one(y, context, est):
    """Hp: the first frozen proposal (local 4x search around the paths and the residual peak, then the residual peak with
    a matched-filter delay) is added and the model refitted, without the global-stop or path-test decision."""
    model = ObservationModel(context)
    cfg = context.config
    r = y - model.mean(est)
    centers = [(est.aod[k], est.aoa[k]) for k in range(est.num_paths)] + [VL._residual_peak(model, r, context)]
    loc, _ = VL.local_candidates(model, context, r, centers)
    if loc:
        _, tau, aod, aoa, g = loc[0][:5]
    else:
        aod, aoa = VL._residual_peak(model, r, context)
        bb = (model.tx_projection(aod)[:, :, None] * model.rx_response(aoa)[:, None, :]).reshape(model.N, -1)
        coef = np.einsum("nm,nm->n", bb.conj(), r.reshape(model.N, -1)) / np.maximum(np.sum(np.abs(bb) ** 2, axis=1), 1e-30)
        tau = delay_estimation.periodogram_delay(coef, cfg.physics.sampling_period, cfg.estimator.delay_oversampling)
        g = None
    if g is None:
        g = ls_gains(model, r, np.array([tau]), np.array([aod]), np.array([aoa]))[0]
    e = _refit(y, context, ChannelEstimate(np.r_[est.delay, tau], np.r_[est.aod, aod], np.r_[est.aoa, aoa], np.r_[est.gain_tilde, g]))
    return e, _fit(model, y, e)


def minus_one(y, context, est):
    """Hm: remove the path whose removal increases the residual energy least, refit."""
    model = ObservationModel(context)
    if est.num_paths < 2:
        return est, _fit(model, y, est)
    costs = []
    for k in range(est.num_paths):
        sub = est.subset([j for j in range(est.num_paths) if j != k])
        costs.append(_fit(model, y, ChannelEstimate(sub.delay, sub.aod, sub.aoa, ls_gains(model, y, sub.delay, sub.aod, sub.aoa))))
    k = int(np.argmin(costs))
    sub = est.subset([j for j in range(est.num_paths) if j != k])
    e = _refit(y, context, ChannelEstimate(sub.delay, sub.aod, sub.aoa, ls_gains(model, y, sub.delay, sub.aod, sub.aoa)))
    return e, _fit(model, y, e)


def same_support(a, b, context):
    if a.num_paths != b.num_paths:
        return False
    cfg = context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    used = set()
    for k in range(a.num_paths):
        hit = [j for j in range(b.num_paths) if j not in used and abs(a.delay[k] - b.delay[j]) <= Ts
               and abs(np.sin(a.aoa[k]) - np.sin(b.aoa[j])) <= 2.0 / nr and abs(np.sin(a.aod[k]) - np.sin(b.aod[j])) <= 2.0 / nt]
        if not hit:
            return False
        used.add(hit[0])
    return True


def generate(cand, y, context, which=("H0", "H1", "H2", "Hp", "Hm")):
    """{name: (est, f, n0, seconds)} for a B2f3Rv2-like candidate `cand` (its noise mode and networks); context.n0 must
    already be the noise input (plug-in for estimated-noise modes). Hp/Hm/H2 use the noise level of H0."""
    out = {}
    t0 = time.perf_counter()
    e0, f0, n00, log = cand.paths(y, context)
    out["H0"] = (e0, f0, n00, time.perf_counter() - t0, log)
    import dataclasses
    from ..b2f3r_v2.path_recovery import model_order_loop
    from ..models.path_init import network_input
    ctx0 = dataclasses.replace(context, n0=n00)
    if "H1" in which:
        t0 = time.perf_counter()
        e, f, n0, lg = cand.paths(y, context, force_coarse=True)
        out["H1"] = (e, f, n0, time.perf_counter() - t0, lg)
    if "H2" in which:
        t0 = time.perf_counter()
        lg = {}
        e, f, n0 = model_order_loop(cand.frozen.pe, y, context, network_input(y, context), it_mid=cand.frozen.it_mid,
                                    noise=cand.noise, cand_fn=LG.cf_cgs, log=lg)
        out["H2"] = (e, f, n0, time.perf_counter() - t0, lg)
    if "Hp" in which:
        t0 = time.perf_counter()
        e, f = plus_one(y, ctx0, e0)
        out["Hp"] = (e, f, n00, time.perf_counter() - t0, {})
    if "Hm" in which:
        t0 = time.perf_counter()
        e, f = minus_one(y, ctx0, e0)
        out["Hm"] = (e, f, n00, time.perf_counter() - t0, {})
    return out

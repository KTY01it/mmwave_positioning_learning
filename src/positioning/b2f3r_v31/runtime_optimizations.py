"""Exact runtime optimizations of the v3 hypothesis generators (identical outputs; tests/test_b2f3r_v31.py).

O-lazy  `local_best`: the frozen `models.v3_local.local_candidates` computes the grid statistic of every window, then
        refines (off-grid search, delay, gain) EVERY window, and finally sorts by the *grid* statistic; `plus_one` uses only
        the first element. Refining only the window with the largest grid statistic (first on ties, as the stable sort)
        gives the identical proposal.
O-dict  steering responses of the local window grids are computed once per window in the frozen code as well; no further
        exact reuse is available without changing the order of floating-point operations.
"""
from __future__ import annotations

import numpy as np

from ..models import v3_local as VL
from ..models.path_init import ls_gains
from ..models.v3_aosage import _fit
from ..signal_processing import beamspace, delay_estimation
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..b2f3r_v3.hypotheses import _refit


def local_best(model, context, r, centers):
    """(stat, tau, aod, aoa, gain) of loc[0] of VL.local_candidates(model, context, r, centers), refining one window."""
    cfg = context.config
    e = cfg.estimator
    nt, nr = cfg.array.num_tx, cfg.array.num_rx
    Ts = cfg.physics.sampling_period
    R3 = r.reshape(model.N, model.G, model.nr)
    yb = r.reshape(model.N, -1)
    lam_c, d = cfg.physics.wavelength_c, cfg.antenna_spacing
    best = None
    for aod0, aoa0 in centers:
        st0, sr0 = np.sin(aod0), np.sin(aoa0)
        dst = np.arange(-VL.HALF_CELLS * VL.OS_LOCAL, VL.HALF_CELLS * VL.OS_LOCAL + 1) * 2.0 / (nt * VL.OS_LOCAL)
        dsr = np.arange(-VL.HALF_CELLS * VL.OS_LOCAL, VL.HALF_CELLS * VL.OS_LOCAL + 1) * 2.0 / (nr * VL.OS_LOCAL)
        aods = beamspace.sine_to_angle(np.clip(st0 + dst, -1, 1), e.aod_branch)
        aoas = beamspace.sine_to_angle(np.clip(sr0 + dsr, -1, 1), e.aoa_branch)
        tx = model.tx_projection(np.asarray(aods))
        rx = model.rx_response(np.asarray(aoas))
        W = np.einsum("jnr,ngr->jng", rx.conj(), R3)
        num = np.abs(np.einsum("ing,jng->ijn", tx.conj(), W)) ** 2
        den = np.einsum("in,jn->ijn", np.sum(np.abs(tx) ** 2, axis=2), np.sum(np.abs(rx) ** 2, axis=2))
        stat = np.sum(num / np.maximum(den, 1e-30), axis=2)
        i, j = np.unravel_index(int(np.argmax(stat)), stat.shape)
        if best is None or float(stat[i, j]) > best[0]:
            best = (float(stat[i, j]), aods, aoas, i, j)
    s, aods, aoas, i, j = best
    res_t = VL.angle_resolution(aods[i], nt, lam_c, d) / VL.OS_LOCAL
    res_r = VL.angle_resolution(aoas[j], nr, lam_c, d) / VL.OS_LOCAL
    aod, aoa = VL._refine_atom(model, yb, float(aods[i]), float(aoas[j]), 1.2 * res_t, 1.2 * res_r)
    b = (model.tx_projection(aod)[:, :, None] * model.rx_response(aoa)[:, None, :]).reshape(model.N, -1)
    coef = np.einsum("nm,nm->n", b.conj(), yb) / np.maximum(np.sum(np.abs(b) ** 2, axis=1), 1e-30)
    tau = delay_estimation.periodogram_delay(coef, Ts, e.delay_oversampling)
    g = delay_estimation.gain_from_delay(coef, tau, 1.0, Ts)
    return s, tau, aod, aoa, g


def plus_one_fast(y, context, est, tm=None):
    """Identical to b2f3r_v3.hypotheses.plus_one with `local_best`; tm receives 'search' and 'refit' seconds."""
    import time
    tm = {} if tm is None else tm
    t0 = time.perf_counter()
    model = ObservationModel(context)
    r = y - model.mean(est)
    centers = [(est.aod[k], est.aoa[k]) for k in range(est.num_paths)] + [VL._residual_peak(model, r, context)]
    _, tau, aod, aoa, g = local_best(model, context, r, centers)
    tm["search"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    e = _refit(y, context, ChannelEstimate(np.r_[est.delay, tau], np.r_[est.aod, aod], np.r_[est.aoa, aoa], np.r_[est.gain_tilde, g]))
    f = _fit(model, y, e)
    tm["refit"] = time.perf_counter() - t0
    return e, f

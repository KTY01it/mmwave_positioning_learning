"""V3-2B0 (docs/v3_2b0_protocol.md, locked): local multipath recovery inside the adaptive model-order SAGE of V3-1/A3.

Variants: B1 = A3 without proximity exclusions; B2 = B1 + local 4x off-grid search on the residual (DCS-SOMP statistic
(42) on the 4x grid in +-2-cell windows around every current path and the residual peak, off-grid refinement
`_refine_atom` (F17), periodogram delay (54), gain (53)) as first proposal source; B3 = B2 + joint nonlinear least
squares over all path parameters before every GLRT decision. GLRT z, global stop and N0 unchanged (A3). No strong
classical output is used; only its algorithms [LIT, Shahmansoori et al. TWC 2018], activated locally [DESIGN].
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import chi2

from ..signal_processing import beamspace, delay_estimation
from ..signal_processing.coarse import _refine_atom
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import angle_resolution, sage
from .path_init import PathInitEstimator, decode, ls_gains, network_input
from .v3_aosage import L_MAX, R_MAX, Z_DELTA, _fit

OS_LOCAL, HALF_CELLS = 4, 2


def _residual_peak(model, r, context):
    """Strongest cell of the residual angle-delay map (1x unitary grid), NO masking (B1-B3)."""
    from .features import correlation_cube
    from .path_init import index_to_sine
    cfg = context.config
    c = correlation_cube(r, context)
    N = c.shape[0]
    P = np.abs(np.fft.ifft(c, axis=0) * np.sqrt(N)) ** 2
    b, it, ir = np.unravel_index(int(np.argmax(P)), P.shape)
    e = cfg.estimator
    nt, nr = cfg.array.num_tx, cfg.array.num_rx
    return (float(beamspace.sine_to_angle(float(np.clip(index_to_sine(it, nt), -1, 1)), e.aod_branch)),
            float(beamspace.sine_to_angle(float(np.clip(index_to_sine(ir, nr), -1, 1)), e.aoa_branch)))


def local_candidates(model, context, r, centers):
    """Local 4x DCS-SOMP statistic on residual r around each (aod, aoa) centre; best grid point per window -> off-grid
    refinement -> delay (54) and gain (53). Returns (list of (stat, tau, aod, aoa, gain), number of atoms evaluated)."""
    cfg = context.config
    e = cfg.estimator
    nt, nr, N = cfg.array.num_tx, cfg.array.num_rx, cfg.physics.num_subcarriers
    Ts = cfg.physics.sampling_period
    R3 = r.reshape(model.N, model.G, model.nr)
    yb = r.reshape(model.N, -1)
    out, n_atoms = [], 0
    lam_c, d = cfg.physics.wavelength_c, cfg.antenna_spacing
    for aod0, aoa0 in centers:
        st0, sr0 = np.sin(aod0), np.sin(aoa0)
        dst = np.arange(-HALF_CELLS * OS_LOCAL, HALF_CELLS * OS_LOCAL + 1) * 2.0 / (nt * OS_LOCAL)
        dsr = np.arange(-HALF_CELLS * OS_LOCAL, HALF_CELLS * OS_LOCAL + 1) * 2.0 / (nr * OS_LOCAL)
        aods = beamspace.sine_to_angle(np.clip(st0 + dst, -1, 1), e.aod_branch)
        aoas = beamspace.sine_to_angle(np.clip(sr0 + dsr, -1, 1), e.aoa_branch)
        tx = model.tx_projection(np.asarray(aods))                 # [I, N, G]
        rx = model.rx_response(np.asarray(aoas))                   # [J, N, Nr]
        W = np.einsum("jnr,ngr->jng", rx.conj(), R3)                # [J, N, G]
        num = np.abs(np.einsum("ing,jng->ijn", tx.conj(), W)) ** 2
        den = np.einsum("in,jn->ijn", np.sum(np.abs(tx) ** 2, axis=2), np.sum(np.abs(rx) ** 2, axis=2))
        stat = np.sum(num / np.maximum(den, 1e-30), axis=2)        # Eq. (42) statistic, per-subcarrier energy
        n_atoms += stat.size
        i, j = np.unravel_index(int(np.argmax(stat)), stat.shape)
        res_t = angle_resolution(aods[i], nt, lam_c, d) / OS_LOCAL
        res_r = angle_resolution(aoas[j], nr, lam_c, d) / OS_LOCAL
        aod, aoa = _refine_atom(model, yb, float(aods[i]), float(aoas[j]), 1.2 * res_t, 1.2 * res_r)
        b = (model.tx_projection(aod)[:, :, None] * model.rx_response(aoa)[:, None, :]).reshape(model.N, -1)
        coef = np.einsum("nm,nm->n", b.conj(), yb) / np.maximum(np.sum(np.abs(b) ** 2, axis=1), 1e-30)
        tau = delay_estimation.periodogram_delay(coef, Ts, e.delay_oversampling)
        g = delay_estimation.gain_from_delay(coef, tau, 1.0, Ts)
        out.append((float(stat[i, j]), tau, aod, aoa, g))
    out.sort(key=lambda q: -q[0])
    return out, n_atoms


def joint_refine(y, context, est, max_nfev: int = 20):
    """Joint nonlinear LS over all (tau, aod, aoa, Re h, Im h) with the analytic Jacobian of the observation model."""
    model = ObservationModel(context)
    K = est.num_paths

    def unpack(v):
        v = v.reshape(K, 5)
        return ChannelEstimate(v[:, 0], v[:, 1], v[:, 2], v[:, 3] + 1j * v[:, 4])

    def fun(v):
        r = (y - model.mean(unpack(v))).ravel()
        return np.r_[r.real, r.imag]

    def jac(v):
        D = model.jacobian(unpack(v))
        return -np.r_[D.real, D.imag]

    v0 = np.stack([est.delay, est.aod, est.aoa, est.gain_tilde.real, est.gain_tilde.imag], axis=1).ravel()
    sc = np.tile([context.config.physics.sampling_period, 1e-2, 1e-2, 1.0, 1.0], K) * np.maximum(1, np.abs(v0) * 0 + 1)
    try:
        res = least_squares(fun, v0, jac=jac, method="lm", max_nfev=max_nfev, x_scale=sc)
        out = unpack(res.x)
        return out if _fit(model, y, out) <= _fit(model, y, est) else est
    except (np.linalg.LinAlgError, ValueError):
        return est


def fold_branches(est: ChannelEstimate, cfg) -> ChannelEstimate:
    """Map every AoD / AoA onto the configured array branch (F9/F27: AoD front, AoA back half-plane). The ULA response
    depends on sin(theta) only, so the observation and the likelihood are unchanged; this only enforces the frozen
    half-plane convention that the geometry stage relies on (V3-2C1: B2's 1-D refinements crossed endfire)."""
    e = cfg.estimator
    return ChannelEstimate(est.delay, np.asarray(beamspace.sine_to_angle(np.sin(est.aod), e.aod_branch), float),
                           np.asarray(beamspace.sine_to_angle(np.sin(est.aoa), e.aoa_branch), float),
                           est.gain_tilde, dict(est.info))


def aosage3(pe: PathInitEstimator, y, context, variant: str = "B2", it_mid: int = 40, delta: float = 1e-8,
            pfa: float = 0.01, x=None):
    """variant B2f = B2 + fold_branches after every SAGE / refinement (V3-2C1 fix)."""
    fold = variant.endswith("f")
    variant = variant[:-1] if fold else variant
    assert variant in ("B1", "B2", "B3")
    cfg = context.config
    Ts = cfg.physics.sampling_period
    model = ObservationModel(context)
    tol = cfg.estimator.sage_tol
    t0 = time.perf_counter()
    x = network_input(y, context) if x is None else x
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    cand = [(p, aod, aoa, tau * Ts) for p, aod, aoa, tau in decode(out, cfg, 0.05, 8)]
    conf = [c for c in cand if c[0] >= pe.threshold] or cand[:1]
    pool = [c for c in cand if c[0] < pe.threshold]
    d, a, b = (np.array([c[i] for c in conf]) for i in (3, 1, 2))
    est = sage(y, context, ChannelEstimate(d, a, b, ls_gains(model, y, d, a, b)), max_iter=it_mid, tol=tol).estimate
    if variant == "B3":
        est = joint_refine(y, context, est)
    if fold:
        est = fold_branches(est, cfg)
    f = _fit(model, y, est)
    s2 = context.n0
    z = Z_DELTA[delta]
    eps = 0.5 * chi2.ppf(1 - pfa, 2 * y.size)
    st = {"local_calls": 0, "local_atoms": 0, "t_local": 0.0, "accepted": 0, "removed": 0, "accepted_local": 0}

    def refit(e):
        e = sage(y, context, e, max_iter=it_mid, tol=tol).estimate
        e = joint_refine(y, context, e) if variant == "B3" else e
        return fold_branches(e, cfg) if fold else e

    def proposals(est):
        """Ordered proposals for the next path: local 4x search (B2/B3), then network pool, then residual peak."""
        r = y - model.mean(est)
        props = []
        if variant in ("B2", "B3"):
            t1 = time.perf_counter()
            centers = [(est.aod[k], est.aoa[k]) for k in range(est.num_paths)] + [_residual_peak(model, r, context)]
            loc, n = local_candidates(model, context, r, centers)
            st["local_calls"] += 1
            st["local_atoms"] += n
            st["t_local"] += time.perf_counter() - t1
            props += [("local", q[1], q[2], q[3], q[4]) for q in loc]
        for c in pool:
            props.append(("pool", c[3], c[1], c[2], None))
        rp = _residual_peak(model, r, context)
        props.append(("peak", None, rp[0], rp[1], None))
        return props

    def forward(est, f):
        changed = False
        while est.num_paths < L_MAX:
            if f / s2 < eps:
                break
            best = None
            r = y - model.mean(est)
            for src, tau, aod, aoa, g in proposals(est)[:R_MAX + (est.num_paths + 1 if variant != "B1" else 0)]:
                if tau is None:              # pool / peak proposals: delay from the matched filter of the residual
                    bb = (model.tx_projection(aod)[:, :, None] * model.rx_response(aoa)[:, None, :]).reshape(model.N, -1)
                    rb = r.reshape(model.N, -1)
                    coef = np.einsum("nm,nm->n", bb.conj(), rb) / np.maximum(np.sum(np.abs(bb) ** 2, axis=1), 1e-30)
                    tau = delay_estimation.periodogram_delay(coef, Ts, cfg.estimator.delay_oversampling)
                if g is None:
                    g = ls_gains(model, r, np.array([tau]), np.array([aod]), np.array([aoa]))[0]
                trial = refit(ChannelEstimate(np.r_[est.delay, tau], np.r_[est.aod, aod], np.r_[est.aoa, aoa],
                                              np.r_[est.gain_tilde, g]))
                f_new = _fit(model, y, trial)
                if (f - f_new) / s2 > z:
                    best = (trial, f_new, src)
                    break
            if best is None:
                break
            est, f = best[0], best[1]
            st["accepted"] += 1
            st["accepted_local"] += best[2] == "local"
            changed = True
        return est, f, changed

    def backward(est, f):
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
            if (costs[k] - f_all) / s2 > z:
                break
            est = est.subset([j for j in range(est.num_paths) if j != k])
            st["removed"] += 1
            changed = True
        if changed:
            est = refit(ChannelEstimate(est.delay, est.aod, est.aoa, ls_gains(model, y, est.delay, est.aod, est.aoa)))
            f = _fit(model, y, est)
        return est, f, changed

    for _ in range(3):
        est, f, c1 = forward(est, f)
        est, f, c2 = backward(est, f)
        if not (c1 or c2):
            break
    est.info.update({"v3local": {**st, "t": time.perf_counter() - t0, "variant": variant}})
    return est

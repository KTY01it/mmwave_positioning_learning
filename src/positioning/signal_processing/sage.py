"""Step 2 (Sec. IV-C): SAGE refinement, Eqs. (55)-(60), Gauss-Seidel coordinate updates.

E-step (59): z_k = y - sum_{l != k} mu(eta_l).
M-step (60): maximise Q(eta_k) = -||z_k - mu(eta_k)||^2 coordinate-wise over theta_Tx, theta_Rx, tau,
then h~ in closed form. With h~ concentrated out, each 1D problem maximises |b(x)^H z_k|^2 / ||b(x)||^2.
Runs in the element domain, which is equivalent to the beamspace formulation for unitary U (F13).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .model import ChannelEstimate, ObservationModel


def maximize_1d(f_vec, x0: float, half_width: float, num_grid: int = 17, xatol: float = 1e-12) -> float:
    """Local maximiser: dense grid on [x0-hw, x0+hw], then bounded Brent around the best grid point."""
    xs = x0 + np.linspace(-half_width, half_width, num_grid)
    v = f_vec(xs)
    i = int(np.argmax(v))
    lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, num_grid - 1)]
    res = minimize_scalar(lambda x: -f_vec(np.array([x]))[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": xatol})
    return float(res.x) if -res.fun >= v[i] else float(xs[i])


def angle_resolution(theta, num_antennas, wavelength_c, spacing):
    """Width of one dictionary bin in theta: (lambda_c / d) / Nant / |cos theta|."""
    return (wavelength_c / spacing) / num_antennas / max(abs(np.cos(theta)), 0.05)


# --------------------------------------------------------------------------------------
# concentrated objectives for one path given its "hidden data" z [N, G, Nr]
# --------------------------------------------------------------------------------------
def objective_aod(model: ObservationModel, z, tau, aoa):
    e = model.delay_phasor(tau)
    W = np.conj(e)[:, None] * np.einsum("nr,ngr->ng", model.rx_response(aoa).conj(), z)

    def f(thetas):
        tp = model.tx_projection(np.asarray(thetas))  # [C, N, G]
        num = np.abs(np.einsum("cng,ng->c", tp.conj(), W)) ** 2
        return num / np.sum(np.abs(tp) ** 2, axis=(1, 2))
    return f


def objective_aoa(model: ObservationModel, z, tau, aod):
    e = model.delay_phasor(tau)
    V = np.conj(e)[:, None] * np.einsum("ng,ngr->nr", model.tx_projection(aod).conj(), z)

    def f(thetas):
        ar = model.rx_response(np.asarray(thetas))  # [C, N, Nr]
        return np.abs(np.einsum("cnr,nr->c", ar.conj(), V)) ** 2
    return f


def objective_delay(model: ObservationModel, z, aod, aoa):
    Q = np.einsum("ng,nr,ngr->n", model.tx_projection(aod).conj(), model.rx_response(aoa).conj(), z)

    def f(taus):
        e = model.delay_phasor(np.asarray(taus))  # [C, N]
        return np.abs(e.conj() @ Q) ** 2
    return f


def best_gain(model: ObservationModel, z, tau, aod, aoa) -> complex:
    b = model.path_basis(tau, aod, aoa)
    return complex(np.vdot(b, z) / np.vdot(b, b).real)


# --------------------------------------------------------------------------------------
@dataclass
class SAGEResult:
    estimate: ChannelEstimate
    iterations: int
    converged: bool
    history: list = field(default_factory=list)  # ChannelEstimate after each iteration (index 0 = init)
    residual_energy: list = field(default_factory=list)


def sage(y: np.ndarray, context, init: ChannelEstimate, *, max_iter: int = 20, tol: float = 1e-6,
         record_history: bool = False) -> SAGEResult:
    """Refine all paths. ``tol`` is relative to the resolution (Ts for delay, one bin for angles)."""
    cfg = context.config
    model = ObservationModel(context)
    lam_c, d = cfg.physics.wavelength_c, cfg.antenna_spacing
    Ts = cfg.physics.sampling_period
    est = ChannelEstimate(init.delay.astype(float).copy(), init.aod.astype(float).copy(),
                          init.aoa.astype(float).copy(), init.gain_tilde.astype(complex).copy(),
                          dict(init.info))
    K = est.num_paths
    mus = [model.path_signal(est.delay[k], est.aod[k], est.aoa[k], est.gain_tilde[k]) for k in range(K)]
    history = [ChannelEstimate(est.delay.copy(), est.aod.copy(), est.aoa.copy(), est.gain_tilde.copy())] \
        if record_history else []
    res_energy = [float(np.sum(np.abs(y - sum(mus)) ** 2))] if K else []
    # search half-widths: one resolution cell at first, then adaptive
    hw_t = np.array([angle_resolution(est.aod[k], cfg.array.num_tx, lam_c, d) for k in range(K)])
    hw_r = np.array([angle_resolution(est.aoa[k], cfg.array.num_rx, lam_c, d) for k in range(K)])
    hw_d = np.full(K, Ts)
    res_t, res_r = hw_t.copy(), hw_r.copy()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        max_change = 0.0
        total = sum(mus) if K else 0
        for k in range(K):
            z = y - (total - mus[k])  # E-step, Eq. (59)
            old = (est.delay[k], est.aod[k], est.aoa[k])
            est.aod[k] = maximize_1d(objective_aod(model, z, est.delay[k], est.aoa[k]), est.aod[k], hw_t[k])
            est.aoa[k] = maximize_1d(objective_aoa(model, z, est.delay[k], est.aod[k]), est.aoa[k], hw_r[k])
            est.delay[k] = maximize_1d(objective_delay(model, z, est.aod[k], est.aoa[k]), est.delay[k], hw_d[k])
            est.gain_tilde[k] = best_gain(model, z, est.delay[k], est.aod[k], est.aoa[k])
            new_mu = model.path_signal(est.delay[k], est.aod[k], est.aoa[k], est.gain_tilde[k])
            total = total - mus[k] + new_mu
            mus[k] = new_mu
            ch = np.array([abs(est.delay[k] - old[0]) / Ts, abs(est.aod[k] - old[1]) / res_t[k],
                           abs(est.aoa[k] - old[2]) / res_r[k]])
            max_change = max(max_change, float(ch.max()))
            # adapt windows: never smaller than 5 % of a cell, never larger than one cell
            hw_d[k] = np.clip(4 * abs(est.delay[k] - old[0]), 0.05 * Ts, Ts)
            hw_t[k] = np.clip(4 * abs(est.aod[k] - old[1]), 0.05 * res_t[k], res_t[k])
            hw_r[k] = np.clip(4 * abs(est.aoa[k] - old[2]), 0.05 * res_r[k], res_r[k])
        if record_history:
            history.append(ChannelEstimate(est.delay.copy(), est.aod.copy(), est.aoa.copy(), est.gain_tilde.copy()))
        res_energy.append(float(np.sum(np.abs(y - total) ** 2)))
        if max_change < tol:
            converged = True
            break
    est.info["sage_iterations"] = it
    return SAGEResult(est, it, converged, history, res_energy)

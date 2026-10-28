"""Extended invariance principle (EXIP) solved with Levenberg-Marquardt, Eqs. (63)-(64).

v(eta_bar) = (eta_hat - f(eta_bar))^T W (eta_hat - f(eta_bar)),  W = J_eta (FIM at the estimates) or a
unit-normalised identity. Angle residuals are wrapped to (-pi, pi].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..bounds.fim import channel_fim
from ..signal_processing.model import ChannelEstimate
from ..system import geometry


@dataclass
class EXIPResult:
    position: np.ndarray
    orientation: float
    scatterers: np.ndarray
    gains: np.ndarray
    cost: float
    x: np.ndarray
    success: bool
    nfev: int
    los_index: int | None = None  # NLOS: index (in the input estimate) of the path used as LOS
    kept_paths: list | None = None  # NLOS with pruning: indices of the paths kept


def weight_matrix(est: ChannelEstimate, context, weighting: str) -> np.ndarray:
    """W for Eq. (63). 'fim': J_eta at eta_hat. 'identity': identity after expressing tau as c*tau [m]
    and gains relative to |h~_k| (otherwise an identity in SI units would ignore the delays)."""
    if weighting == "fim":
        return channel_fim(context, est)
    if weighting == "identity":
        c = context.config.physics.speed_of_light
        g = np.maximum(np.abs(est.gain_tilde), 1e-300)
        return np.diag(np.stack([np.full(est.num_paths, c ** 2), np.ones(est.num_paths), np.ones(est.num_paths),
                                 1 / g ** 2, 1 / g ** 2], axis=1).ravel())
    raise ValueError(f"unknown weighting {weighting!r}")


def _sqrt_weight(W: np.ndarray) -> np.ndarray:
    """L^T with W = L L^T (eigen-decomposition fallback for semi-definite W)."""
    Ws = 0.5 * (W + W.T)
    try:
        return np.linalg.cholesky(Ws).T
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Ws)
        return (V * np.sqrt(np.clip(w, 0, None))).T


def residual_vector(eta_hat: np.ndarray, eta_model: np.ndarray) -> np.ndarray:
    r = (eta_hat - eta_model).reshape(-1, 5).copy()
    r[:, 1:3] = geometry.wrap_angle(r[:, 1:3])
    return r.ravel()


def unpack(x: np.ndarray, num_paths: int, has_los: bool):
    """x follows geometry.geometry_parameter_layout -> (p, alpha, scatterers [S,2], gains [K])."""
    p, alpha = x[0:2], x[2]
    gains, scat = [], []
    i = 3
    for k in range(num_paths):
        if has_los and k == 0:
            gains.append(x[i] + 1j * x[i + 1]); i += 2
        else:
            scat.append(x[i:i + 2]); gains.append(x[i + 2] + 1j * x[i + 3]); i += 4
    return p, float(alpha), np.array(scat).reshape(-1, 2), np.array(gains)


def pack(p, alpha, scatterers, gains, has_los: bool) -> np.ndarray:
    x = [p[0], p[1], alpha]
    scatterers = np.asarray(scatterers).reshape(-1, 2)
    j = 0
    for k, g in enumerate(gains):
        if has_los and k == 0:
            x += [g.real, g.imag]
        else:
            x += [scatterers[j, 0], scatterers[j, 1], g.real, g.imag]; j += 1
    return np.array(x, dtype=float)


def model_eta(x, num_paths, has_los, bs_position, c) -> np.ndarray:
    p, alpha, scat, gains = unpack(x, num_paths, has_los)
    g = geometry.compute_path_geometry(bs_position, p, alpha, scat, has_los, c)
    return np.stack([g.delay, g.aod, g.aoa, gains.real, gains.imag], axis=1).ravel()


def exip_lma(est: ChannelEstimate, W: np.ndarray, x0: np.ndarray, has_los: bool, bs_position, c,
             max_nfev: int = 2000) -> EXIPResult:
    """Minimise Eq. (63)/(64) from x0. Paths of ``est`` must be ordered as the layout (LOS first)."""
    eta_hat = est.as_eta()
    Lt = _sqrt_weight(W)
    K = est.num_paths

    def fun(x):
        return Lt @ residual_vector(eta_hat, model_eta(x, K, has_los, bs_position, c))

    def jac(x):  # d f / d x^T = T^T (Eq. 16); the layouts coincide by construction
        p, alpha, scat, _ = unpack(x, K, has_los)
        return -Lt @ geometry.transformation_matrix(bs_position, p, alpha, scat, has_los, c).T

    method = "lm" if Lt.shape[0] >= x0.size else "trf"
    try:
        sol = least_squares(fun, x0, jac=jac, method=method, x_scale="jac", max_nfev=max_nfev)
        x, ok, nfev = sol.x, bool(sol.success), int(sol.nfev)
    except Exception:  # degenerate geometry (e.g. parallel lines) -> keep the initial point
        x, ok, nfev = x0, False, 0
    r = fun(x)
    p, alpha, scat, gains = unpack(x, K, has_los)
    return EXIPResult(p, float(geometry.wrap_angle(alpha)), scat, gains, float(r @ r), x, ok, nfev)

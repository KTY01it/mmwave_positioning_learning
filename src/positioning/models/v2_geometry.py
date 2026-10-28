"""v2 geometry solvers (V2-B, docs/v2_protocol.md): explicit geometric solvers in which any learned part predicts only
residual directions / weights, never absolute coordinates.

Classical core, used by every variant (OLOS):
  Shahmansoori Eq. (65), F16 convention, stacked over ALL K paths [ADAPTATION: the paper / repo use one pair]:
      p - d_k (u_T,k + u_R,k) = q - c tau_k u_R,k,   u_T,k = u(theta_Tx,k), u_R,k = u(theta_Rx,k + alpha)
  i.e. h(alpha) = G(alpha) x with x = [p, d_1..d_K] (2K equations, 2 + K unknowns), solved by weighted LS (23) of
  J. Yang et al., "Model-Based Learning Network for 3-D Localization in mmWave Communications", IEEE TWC 20(8), 2021,
  with the first-order weight W = (B Q B^T)^-1 (their Eqs. (24)-(26), our B derived for Eq. (65)):
      e_k ~ -c dtau_k u_R,k + d_k dthTx_k u_perp(theta_Tx,k) - (c tau_k - d_k) dthRx_k u_perp(theta_Rx,k + alpha)
  Q = CRB block of (tau, theta_Tx, theta_Rx) from the inverse FIM of the channel parameters; iterated (their Alg. 1(b)).
  alpha: explicit grid; the trial is selected by the EXIP cost (64) of the linear solution; then ONE LMA (EXIP, W = FIM).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..localization.exip_lma import _sqrt_weight, exip_lma, model_eta, pack, residual_vector, weight_matrix
from ..localization.pipeline import localize
from ..signal_processing.model import ObservationModel
from ..system import geometry

ALPHAS = np.arange(-0.5, 0.5 + 1e-12, 0.05)


def _uperp(phi):
    return np.stack([-np.sin(phi), np.cos(phi)], axis=-1)


def stacked_system(e, alpha, q, c):
    K = e.num_paths
    A = np.zeros((2 * K, 2 + K))
    b = np.zeros(2 * K)
    uT, uR = geometry.unit(e.aod), geometry.unit(e.aoa + alpha)
    for k in range(K):
        r = slice(2 * k, 2 * k + 2)
        A[r, 0:2] = np.eye(2)
        A[r, 2 + k] = -(uT[k] + uR[k])
        b[r] = q - c * e.delay[k] * uR[k]
    return A, b


def first_order_B(e, alpha, d, c):
    """[2K, 3K] block-diagonal: d e_k / d(tau_k, thTx_k, thRx_k)."""
    K = e.num_paths
    B = np.zeros((2 * K, 3 * K))
    uR = geometry.unit(e.aoa + alpha)
    for k in range(K):
        r = slice(2 * k, 2 * k + 2)
        B[r, 3 * k] = -c * uR[k]
        B[r, 3 * k + 1] = d[k] * _uperp(e.aod[k])
        B[r, 3 * k + 2] = -(c * e.delay[k] - d[k]) * _uperp(e.aoa[k] + alpha)
    return B


def channel_crb(e, context, W_fim):
    """Q: CRB of (tau, thTx, thRx) per path (marginal, from the inverse FIM incl. gains; 5 parameters per path)."""
    K = e.num_paths
    try:
        C = np.linalg.inv(W_fim)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(W_fim)
    idx = np.array([[5 * k, 5 * k + 1, 5 * k + 2] for k in range(K)]).ravel()
    return C[np.ix_(idx, idx)]


def wls_solve(A, b, W):
    AtW = A.T @ W
    return np.linalg.solve(AtW @ A + 1e-12 * np.eye(A.shape[1]), AtW @ b)


def scatterers_from(x, e, q):
    K = e.num_paths
    return np.stack([q + x[2 + k] * geometry.unit(e.aod[k]) for k in range(K)])


def olos_wls(e, context, residual_dir=None, eps=None, iterations: int = 2):
    """alpha grid -> stacked weighted LS per alpha -> best initial EXIP cost -> one LMA.
    residual_dir: optional callable (e, alpha) -> unit residual direction r_hat in R^{2K} (NN-WLS, Eq. (36) of Yang
    et al.): W <- (Wc^-1 + r r^T / eps)^-1 ... applied in the whitened domain (see nn variant)."""
    cfg = context.config
    c, q = cfg.physics.speed_of_light, np.asarray(context.bs_position, float)
    W_fim = weight_matrix(e, context, cfg.estimator.exip_weighting)
    Q = channel_crb(e, context, W_fim)
    Lt = _sqrt_weight(W_fim)
    eta_hat = e.as_eta()
    K = e.num_paths
    best = None
    for a in ALPHAS:
        try:
            x0 = _wls_trial(e, a, q, c, Q, residual_dir, eps, iterations)
        except np.linalg.LinAlgError:
            continue
        if x0 is None:
            continue
        cost = float(np.sum((Lt @ residual_vector(eta_hat, model_eta(x0, K, False, q, c))) ** 2))
        if best is None or cost < best[0]:
            best = (cost, x0)
    if best is None:
        return None
    return exip_lma(e, W_fim, best[1], False, q, c)


def _wls_trial(e, a, q, c, Q, residual_dir, eps, iterations):
    K = e.num_paths
    A, b = stacked_system(e, a, q, c)
    x = np.linalg.lstsq(A, b, rcond=None)[0]                         # unweighted start (Alg. 1(b), step 0)
    for _ in range(iterations):
        d = np.maximum(x[2:], 1e-3)
        B = first_order_B(e, a, d, c)
        S = B @ Q @ B.T + 1e-12 * np.eye(2 * K)
        if residual_dir is not None:
            r = residual_dir(e, a, x)
            S = S + (r[:, None] * r[None, :]) / (eps if eps is not None else 1.0)  # learned dominant residual
        W = np.linalg.inv(S)
        x = wls_solve(A, b, W)
    p, s = x[:2], scatterers_from(x, e, q)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))):
        return None
    return pack(p, a, s, e.gain_tilde, has_los=False)


def make(name):
    """Geometry solver factory for scripts/31_v2_dev.py: f(est, y, context, cond) -> PositionEstimate with gate scores."""
    from .base import PositionEstimate

    def gate_scores(out, est, y, context):
        ch = out.channel if out.channel is not None else est
        M = y.size
        res = float(np.sum(np.abs(y - ObservationModel(context).mean(ch)) ** 2) / context.n0)
        out.info["gate"] = {"r_norm": res / M, "z": (res - M) / np.sqrt(M), "v": float(out.cost) if np.isfinite(out.cost) else 0.0}
        return out

    if name == "wls":
        def f(est, y, context, cond):
            if cond == "olos" and est.num_paths >= 2:
                e = est.sorted_by_delay()
                r = olos_wls(e, context) if e.num_paths >= 3 else None
                if r is None:   # K = 2: no redundancy (weights have no effect) -> classical best-initial-cost grid, 1 LMA
                    c2 = dataclasses.replace(context, config=context.config.replace(**{"estimator.olos_lma_top_k": 1}))
                    out = localize(est, c2, cond)
                else:
                    out = PositionEstimate(r.position, r.orientation, "olos", r.scatterers, e, r.cost)
            else:
                out = localize(est, context, cond)
            return gate_scores(out, est, y, context)
        return f
    raise KeyError(name)

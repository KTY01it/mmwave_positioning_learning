"""OLOS localisation (LOS blocked), Eqs. (64)-(65): grid over alpha, linear init, EXIP/LMA per trial."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..signal_processing.model import ChannelEstimate
from ..system import geometry
from .exip_lma import EXIPResult, _sqrt_weight, exip_lma, model_eta, pack, residual_vector, weight_matrix


def olos_initial_point(est: ChannelEstimate, alpha, pair, bs_position, c):
    """Solve (65) for the two paths in ``pair``; other scatterers by ray intersection from p."""
    k1, k2 = pair
    p, _, s12 = geometry.olos_linear_solve(bs_position, alpha, est.delay[[k1, k2]], est.aod[[k1, k2]],
                                           est.aoa[[k1, k2]], c)
    scat = np.zeros((est.num_paths, 2))
    scat[k1], scat[k2] = s12
    for k in range(est.num_paths):
        if k not in pair:
            scat[k] = geometry.scatterer_from_lines(bs_position, p, alpha, est.aod[k], est.aoa[k])
    return p, scat


def olos_localize(est: ChannelEstimate, context, weighting: str = "fim", alpha_step: float = 0.05,
                  alpha_max: float = 0.5, pairs: str = "all", top_k: int | None = None) -> EXIPResult:
    """Keep the trial with the smallest cost v_olos. pairs: 'all' tries every path pair for Eq. (65); 'first_two' uses the two smallest delays.

    top_k: run the LMA only from the k trial points with the smallest initial cost (None = all trials).
    """
    cfg = context.config
    c = cfg.physics.speed_of_light
    q = context.bs_position
    e = est.sorted_by_delay()
    K = e.num_paths
    if K < 2:
        raise ValueError("OLOS needs at least two NLOS paths (three for a determined problem)")
    W = weight_matrix(e, context, weighting)
    if pairs not in ("all", "first_two"):
        raise ValueError(f"unknown pairs {pairs!r}")
    pair_list = list(combinations(range(K), 2)) if pairs == "all" else [(0, 1)]  # e is sorted by delay
    alphas = np.arange(-alpha_max, alpha_max + 1e-12, alpha_step)
    starts = []
    for a in alphas:
        for pair in pair_list:
            p0, s0 = olos_initial_point(e, a, pair, q, c)
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(s0)):
                starts.append(pack(p0, a, s0, e.gain_tilde, has_los=False))
    if top_k is not None and len(starts) > top_k:
        Lt = _sqrt_weight(W)
        eta_hat = e.as_eta()
        init_cost = [np.sum((Lt @ residual_vector(eta_hat, model_eta(x, K, False, q, c))) ** 2) for x in starts]
        starts = [starts[i] for i in np.argsort(init_cost)[:top_k]]
    best = None
    for x0 in starts:
        res = exip_lma(e, W, x0, False, q, c)
        if best is None or res.cost < best.cost:
            best = res
    return best

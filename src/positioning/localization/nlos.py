"""NLOS (LOS + scatterers) localisation via EXIP, Eq. (63), with the initialisation of Sec. IV-D."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..signal_processing.model import ChannelEstimate
from ..system import geometry
from .exip_lma import EXIPResult, exip_lma, pack, weight_matrix
from .los import los_localize


def nlos_initial_point(est: ChannelEstimate, bs_position, c):
    """p, alpha from the smallest-delay path; s_k from the intersection of the BS and MS rays."""
    p0, a0 = los_localize(est, bs_position, c, path_index=0)
    scat = [geometry.scatterer_from_lines(bs_position, p0, a0, est.aod[k], est.aoa[k])
            for k in range(1, est.num_paths)]
    return p0, a0, np.array(scat).reshape(-1, 2)


def _nlos_with_los_index(est: ChannelEstimate, context, weighting: str, los_index: int) -> EXIPResult:
    cfg = context.config
    c = cfg.physics.speed_of_light
    order = [los_index] + [k for k in np.argsort(est.delay) if k != los_index]
    e = est.subset(order)
    q = context.bs_position
    p0, a0, s0 = nlos_initial_point(e, q, c)
    if e.num_paths == 1:
        return EXIPResult(p0, a0, np.zeros((0, 2)), e.gain_tilde.copy(), 0.0, np.r_[p0, a0], True, 0,
                          los_index=int(los_index))
    W = weight_matrix(e, context, weighting)
    x0 = pack(p0, a0, s0, e.gain_tilde, has_los=True)
    res = exip_lma(e, W, x0, True, q, c)
    res.los_index = int(los_index)
    return res


def nlos_localize(est: ChannelEstimate, context, weighting: str = "fim", los_selection: str = "min_delay",
                  prune_level: float | None = None) -> EXIPResult:
    """prune_level: if set, NLOS paths are removed greedily while the EXIP cost is rejected by a
    chi-square(K-1) test at this level (F24: a false-alarm path otherwise drags the whole solution)."""
    res = _nlos_select(est, context, weighting, los_selection)
    if prune_level is None or weighting != "fim":
        return res
    keep = list(range(est.num_paths))
    los = res.los_index if res.los_index is not None else int(np.argmin(est.delay))
    while len(keep) > 1 and chi2.sf(res.cost, len(keep) - 1) < prune_level:
        trials = []
        for j in keep:
            if j == los:
                continue
            sub = [k for k in keep if k != j]
            r = _nlos_with_los_index(est.subset(sub), context, weighting, sub.index(los))
            trials.append((r.cost, sub, r))  # all trials keep len(keep) - 1 paths: same dof
        _, keep, res = min(trials, key=lambda t: t[0])
        res.los_index = los
    res.kept_paths = keep
    return res


def _nlos_select(est: ChannelEstimate, context, weighting: str, los_selection: str) -> EXIPResult:
    """Which estimated path is the LOS (F21):
    'min_delay' paper, Sec. IV-D;  'min_cost' every path tried, smallest EXIP cost kept;
    'strongest' largest |h~| — uses the path-loss model (66)-(67): a reflection carries sigma_0^2 P0 < 1.
    With LOS + one scatterer the EXIP problem has a single redundant equation, and a role-swapped geometry
    fits nearly as well when the delays are close, so neither delay nor cost discriminates reliably."""
    if los_selection == "min_delay" or est.num_paths == 1:
        return _nlos_with_los_index(est, context, weighting, int(np.argmin(est.delay)))
    if los_selection == "strongest":
        return _nlos_with_los_index(est, context, weighting, int(np.argmax(np.abs(est.gain_tilde))))
    if los_selection == "min_cost":
        cands = [_nlos_with_los_index(est, context, weighting, k) for k in range(est.num_paths)]
        return min(cands, key=lambda r: r.cost)
    raise ValueError(f"unknown los_selection {los_selection!r}")

"""Multi-start initialisation for SAGE (master plan P6, mandatory strong classical init baseline).

With few beams G the AoD is observed only through G random beam projections, and the AoD likelihood has several
lobes; greedy DCS-SOMP may lock onto a wrong lobe (G0 beam sweep). Multi-start: for the strongest path, take the
M best local maxima of its concentrated AoD likelihood over the whole visible range, run SAGE from each, and return
all refined candidates (the caller selects by likelihood, optionally with a geometric prior).
"""
from __future__ import annotations

import numpy as np

from .model import ChannelEstimate, ObservationModel
from .sage import SAGEResult, objective_aod, sage


def aod_lobe_candidates(model: ObservationModel, y, est: ChannelEstimate, k: int, num: int, oversampling: int = 4):
    """AoDs of the ``num`` largest local maxima of the concentrated AoD likelihood of path k (others removed)."""
    z = y - sum(model.path_signal(est.delay[l], est.aod[l], est.aoa[l], est.gain_tilde[l])
                for l in range(est.num_paths) if l != k)
    grid = np.arcsin(np.linspace(-1, 1, oversampling * model.nt + 1)[1:-1])
    f = objective_aod(model, z, est.delay[k], est.aoa[k])(grid)
    peaks = [i for i in range(1, grid.size - 1) if f[i] >= f[i - 1] and f[i] >= f[i + 1]]
    peaks = sorted(peaks, key=lambda i: -f[i])
    out = [float(est.aod[k])]
    bin_w = 2.0 / model.nt  # one dictionary bin in sin(theta) for d = lambda/2
    for i in peaks:
        if all(abs(np.sin(grid[i]) - np.sin(a)) > bin_w for a in out):
            out.append(float(grid[i]))
        if len(out) >= num:
            break
    return out


def multistart_sage(y, context, coarse: ChannelEstimate, num_starts: int, max_iter: int, tol: float):
    """Refine ``coarse`` from ``num_starts`` AoD lobes of its strongest path. Returns [(SAGEResult, residual)]."""
    model = ObservationModel(context)
    k = int(np.argmax(np.abs(coarse.gain_tilde)))
    results = []
    for aod in aod_lobe_candidates(model, y, coarse, k, num_starts):
        init = ChannelEstimate(coarse.delay.copy(), coarse.aod.copy(), coarse.aoa.copy(), coarse.gain_tilde.copy())
        init.aod[k] = aod
        sr: SAGEResult = sage(y, context, init, max_iter=max_iter, tol=tol)
        results.append((sr, sr.residual_energy[-1]))
    return results

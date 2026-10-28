"""R3: targeted support hypotheses selected from the R1 failure taxonomy (docs/b2f3r_v31/r1_failure_taxonomy.md).

S1' endfire alias flip [DESIGN, adaptation of S1 'lobe replacement']: with half-wavelength spacing the ULA response at
    sin(theta) = +1 and -1 coincide at the carrier frequency and differ only through the subcarrier-dependent wavelength
    (beam squint). R1 shows that the dominant remaining mirrored-OLOS failure is an AoA estimated at the opposite endfire
    alias. For a path with |sin AoA| (or |sin AoD|) >= thr, the alternative with sin -> -sin is formed in the configured
    branch and the support is refitted with the frozen SAGE step. The observation alone (almost) cannot discriminate the
    two; the hypothesis is compared through the frozen likelihood + geometry pipeline (B4). No signal-based resolution of
    the ambiguity is claimed.
P3  bounded backward pruning to the three most significant paths [DESIGN; declared deviation: none of S1-S5 removes
    several spurious paths, which R1 identifies as the dominant failure at 15-25 dB]: keep the three paths whose removal
    increases the residual most (LS gains), SAGE refit. Three paths = the smallest support with which the OLOS branch is
    used.
"""
from __future__ import annotations

import numpy as np

from ..b2f3r_v3.hypotheses import _refit
from ..models.path_init import ls_gains
from ..models.v3_aosage import _fit
from ..signal_processing import beamspace
from ..signal_processing.model import ChannelEstimate, ObservationModel


def endfire_paths(est, which="aoa", thr=0.95):
    v = est.aoa if which == "aoa" else est.aod
    s = np.abs(np.sin(np.asarray(v, float)))
    return [int(k) for k in np.argsort(-s) if s[k] >= thr]


def alias_flip(y, context, est, k, which="aoa"):
    e = context.config.estimator
    aod, aoa = np.array(est.aod, float), np.array(est.aoa, float)
    if which == "aoa":
        aoa[k] = float(beamspace.sine_to_angle(-np.sin(aoa[k]), e.aoa_branch))
    else:
        aod[k] = float(beamspace.sine_to_angle(-np.sin(aod[k]), e.aod_branch))
    model = ObservationModel(context)
    g = ls_gains(model, y, est.delay, aod, aoa)
    h = _refit(y, context, ChannelEstimate(np.array(est.delay, float), aod, aoa, g))
    return h, _fit(model, y, h)


def removal_costs(y, context, est):
    model = ObservationModel(context)
    K = est.num_paths
    out = []
    for k in range(K):
        s = est.subset([j for j in range(K) if j != k])
        out.append(_fit(model, y, ChannelEstimate(s.delay, s.aod, s.aoa, ls_gains(model, y, s.delay, s.aod, s.aoa))))
    return np.array(out)


def prune_to(y, context, est, k=3, costs=None):
    if est.num_paths <= k:
        return None
    model = ObservationModel(context)
    c = removal_costs(y, context, est) if costs is None else costs
    keep = sorted(int(j) for j in np.argsort(-c)[:k])
    s = est.subset(keep)
    h = _refit(y, context, ChannelEstimate(s.delay, s.aod, s.aoa, ls_gains(model, y, s.delay, s.aod, s.aoa)))
    return h, _fit(model, y, h)

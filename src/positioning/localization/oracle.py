"""Oracle diagnostics for G0 (master plan P1.6). Uses ground truth; NOT part of the frozen baseline."""
from __future__ import annotations

import time

import numpy as np

from ..models.base import Estimator, PositionEstimate
from ..signal_processing.beamspace import sine_to_angle
from ..signal_processing.model import ChannelEstimate
from ..signal_processing.sage import sage
from .exip_lma import exip_lma, pack, weight_matrix
from .los import los_localize


def localize_from_geometry(est: ChannelEstimate, context, condition: str, p0, alpha0, scatterers0) -> PositionEstimate:
    """EXIP/LMA from ONE given geometric starting point (no alpha grid, no ray intersections).

    ``est`` must be in layout order (LOS first for NLOS; scatterer order for OLOS). Used by G0 to measure what an
    oracle (or learned, P6-b) geometric initialisation would save."""
    cfg = context.config
    c = cfg.physics.speed_of_light
    if condition == "los":
        p, a = los_localize(est, context.bs_position, c, path_index=0)
        return PositionEstimate(p, a, "los", channel=est)
    has_los = condition == "nlos"
    W = weight_matrix(est, context, cfg.estimator.exip_weighting)
    r = exip_lma(est, W, pack(p0, alpha0, scatterers0, est.gain_tilde, has_los), has_los, context.bs_position, c)
    return PositionEstimate(r.position, r.orientation, condition, r.scatterers, est, r.cost)


class OracleInitEstimator(Estimator):
    """Go/No-Go G0 (P1.6): SAGE initialised at the true parameters plus small perturbation.

    oracle_geometry=True additionally starts EXIP/LMA at the true (p, alpha, s_k) instead of the paper's
    initialisation (NLOS ray intersection / OLOS alpha grid)."""

    requires_oracle = True

    def __init__(self, condition: str = "auto", perturbation: float = 0.25, seed: int = 0,
                 oracle_geometry: bool = False, observable_only: bool = True):
        self.condition = condition
        self.perturbation = perturbation  # in units of one resolution cell (Ts, one dictionary bin)
        self.rng = np.random.default_rng(seed)
        self.oracle_geometry = oracle_geometry
        # observable_only: the ULA response depends on sin(theta) only, so theta and pi - theta are
        # indistinguishable from y (F9, F27). A fair oracle gives the true angles folded to the estimator's branch
        # convention; observable_only=False also leaks the (unobservable) branch.
        self.observable_only = observable_only
        self.name = "oracle_init+sage+exip" + ("(oracle geometry)" if oracle_geometry else "") + \
            ("" if observable_only else "(with branch)")

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        cfg = context.config
        tr = ChannelEstimate.from_paths(gt.paths)
        if self.observable_only:
            e = cfg.estimator
            tr.aod = np.atleast_1d(sine_to_angle(np.sin(tr.aod), e.aod_branch))
            tr.aoa = np.atleast_1d(sine_to_angle(np.sin(tr.aoa), e.aoa_branch))
        K = tr.num_paths
        bin_t = (cfg.physics.wavelength_c / cfg.antenna_spacing) / cfg.array.num_tx / np.maximum(np.abs(np.cos(tr.aod)), 0.05)
        bin_r = (cfg.physics.wavelength_c / cfg.antenna_spacing) / cfg.array.num_rx / np.maximum(np.abs(np.cos(tr.aoa)), 0.05)
        u = lambda: self.rng.uniform(-1, 1, K) * self.perturbation
        init = ChannelEstimate(tr.delay + u() * cfg.physics.sampling_period, tr.aod + u() * bin_t,
                               tr.aoa + u() * bin_r, tr.gain_tilde.copy())
        t0 = time.perf_counter()
        sr = sage(y, context, init, max_iter=cfg.estimator.sage_max_iter, tol=cfg.estimator.sage_tol)
        timing = {"coarse": 0.0, "sage": time.perf_counter() - t0}
        cond = cfg.scenario.condition if self.condition == "auto" else self.condition
        t0 = time.perf_counter()
        if self.oracle_geometry:
            out = localize_from_geometry(sr.estimate, context, cond, gt.position, gt.orientation, gt.scatterers)
        else:
            from .pipeline import localize  # local import: pipeline re-exports this module
            out = localize(sr.estimate, context, cond)
        timing["localize"] = time.perf_counter() - t0
        out.info.update({"sage": sr, "timing": timing})
        return out

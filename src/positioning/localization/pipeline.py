"""Classical estimator of Shahmansoori et al.: DCS-SOMP -> SAGE -> {LOS (61)-(62), NLOS (63), OLOS (64)}."""
from __future__ import annotations

import time

import numpy as np

from ..models.base import Estimator, PositionEstimate
from ..signal_processing.coarse import coarse_estimate
from ..signal_processing.model import ChannelEstimate
from ..signal_processing.multistart import multistart_sage
from ..signal_processing.sage import sage
from .los import los_localize
from .nlos import nlos_localize
from .olos import olos_localize
from .unknown import unknown_condition_localize


def localize(est: ChannelEstimate, context, condition: str) -> PositionEstimate:
    cfg = context.config
    e = cfg.estimator
    c = cfg.physics.speed_of_light
    if est.num_paths == 0:
        return PositionEstimate.failed("no path detected")
    if condition == "los" or (condition == "nlos" and est.num_paths == 1):
        p, a = los_localize(est, context.bs_position, c)
        return PositionEstimate(p, a, "los", channel=est)
    if condition == "nlos":
        r = nlos_localize(est, context, e.exip_weighting, e.los_selection, e.nlos_prune_level)
        return PositionEstimate(r.position, r.orientation, "nlos", r.scatterers, est, r.cost)
    if condition == "olos":
        if est.num_paths < 2:
            return PositionEstimate.failed("OLOS needs >= 2 paths")
        r = olos_localize(est, context, e.exip_weighting, e.olos_alpha_step, e.olos_alpha_max,
                          top_k=e.olos_lma_top_k, pairs=e.olos_pairs)
        return PositionEstimate(r.position, r.orientation, "olos", r.scatterers, est, r.cost)
    if condition == "unknown":
        r, cond, info = unknown_condition_localize(est, context, e.exip_weighting, e.olos_alpha_step,
                                                   e.olos_alpha_max, e.olos_lma_top_k,
                                                   e.unknown_rule, e.unknown_test_level,
                                                   e.los_selection, e.nlos_prune_level, e.olos_pairs)
        return PositionEstimate(r.position, r.orientation, cond, r.scatterers, est, r.cost, info=info)
    raise ValueError(f"unknown condition {condition!r}")


class ClassicalEstimator(Estimator):
    """condition: 'los' | 'nlos' | 'olos' | 'unknown' | 'auto' (= the configured scenario condition,
    i.e. the receiver is told the condition, as in Figs. 3-9)."""

    requires_oracle = False

    def __init__(self, condition: str = "auto", use_sage: bool | None = None, record_history: bool = False):
        self.condition = condition
        self.use_sage = use_sage
        self.record_history = record_history
        self.name = "dcs_somp" + ("+sage" if use_sage is not False else "") + "+exip"

    def channel_estimate(self, y, context):
        cfg = context.config
        t0 = time.perf_counter()
        coarse = coarse_estimate(y, context)
        timing = {"coarse": time.perf_counter() - t0, "sage": 0.0}
        use_sage = cfg.estimator.sage_enabled if self.use_sage is None else self.use_sage
        info = {"coarse": coarse, "timing": timing}
        if use_sage and coarse.num_paths:
            t0 = time.perf_counter()
            sr = sage(y, context, coarse, max_iter=cfg.estimator.sage_max_iter, tol=cfg.estimator.sage_tol,
                      record_history=self.record_history)
            timing["sage"] = time.perf_counter() - t0
            info["sage"] = sr
            return sr.estimate, info
        return coarse, info

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        cfg = context.config
        cond = cfg.scenario.condition if self.condition == "auto" else self.condition
        if cfg.estimator.multistart > 0:
            return self._estimate_multistart(y, context, cond)
        est, info = self.channel_estimate(y, context)
        t0 = time.perf_counter()
        out = localize(est, context, cond)
        info["timing"]["localize"] = time.perf_counter() - t0
        out.info.update(info)
        return out

    def _estimate_multistart(self, y, context, cond) -> PositionEstimate:
        """Multi-start SAGE from several AoD lobes; select by likelihood (optionally inside a geometric prior)."""
        e = context.config.estimator
        t0 = time.perf_counter()
        coarse = coarse_estimate(y, context)
        timing = {"coarse": time.perf_counter() - t0}
        if not coarse.num_paths:
            return PositionEstimate.failed("no path detected")
        t0 = time.perf_counter()
        cands = multistart_sage(y, context, coarse, e.multistart, e.sage_max_iter, e.sage_tol)
        timing["sage"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        outs = []
        for sr, res in sorted(cands, key=lambda c: c[1]):  # best likelihood first
            out = localize(sr.estimate, context, cond)
            outs.append((out, sr, res))
            if e.prior_box is None:
                break  # pure ML: the best-likelihood candidate is the answer
        chosen = outs[0]
        if e.prior_box is not None:
            pb = e.prior_box
            m = pb.get("margin", 0.0)
            lo, hi = np.asarray(pb["p_low"], float) - m, np.asarray(pb["p_high"], float) + m
            inside = [o for o in outs if np.all(np.isfinite(o[0].position)) and np.all(o[0].position >= lo)
                      and np.all(o[0].position <= hi) and abs(o[0].orientation) <= pb["alpha_max"] + m]
            if inside:
                chosen = inside[0]
        timing["localize"] = time.perf_counter() - t0
        out, sr, _ = chosen
        out.info.update({"coarse": coarse, "sage": sr, "timing": timing, "multistart_candidates": len(cands)})
        return out


# Diagnostics that need ground truth live in localization/oracle.py (not part of the frozen baseline).
from .oracle import OracleInitEstimator, localize_from_geometry  # noqa: E402,F401

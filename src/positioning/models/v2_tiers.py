"""V2-C (docs/v2_protocol.md): adaptive refinement + hierarchical fallback on the v1R components (V2-A and V2-B
were not adopted). Condition unknown: learned router (v1R) -> branch.

  tier 1  fast:    v1R path init -> 1 SAGE -> (41) -> LOS closed form / NLOS EXIP / OLOS learned init + 1 LMA
  tier 2  refined: +2 SAGE iterations from the tier-1 channel estimate (3 in total); OLOS: LMA from 3 starts (learned
                   init + the 2 classical alpha-grid starts with the smallest initial EXIP cost), keep the lowest cost
  tier 3  strong classical v2.2, unknown condition (never uses c_hat)
Policy [DESIGN]: accept tier t iff its output is finite (validity, gate v1.1) and its normalised residual
z_t = (||Y - Y_hat||^2 / N0 - M) / sqrt(M) <= th_t. Thresholds (th_1, th_2) per G from calib_v2 by the two-tier
extension of the 6C rule: minimise E[T] subject to P_out and P_cat <= strong + 1 pt in each main SNR band.
Every tier is a [LIT] solver (SAGE: Fleury et al. 1999 / Shahmansoori Sec. IV-C; EXIP-LMA: Sec. IV-D).
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from ..localization.olos import olos_localize
from ..localization.pipeline import ClassicalEstimator, localize
from ..signal_processing.model import ObservationModel
from ..signal_processing.sage import sage
from .base import Estimator, PositionEstimate
from .geom_init import GeomInitializer
from .path_init import PathInitEstimator, network_input
from .router import LearnedRouter, branch_localize, fast_channel_estimate

ROOT = Path(__file__).resolve().parents[3]
V1R = ROOT / "outputs" / "models" / "v1r"


def _z(out, est, y, context):
    ch = out.channel if out.channel is not None else est
    M = y.size
    res = float(np.sum(np.abs(y - ObservationModel(context).mean(ch)) ** 2) / context.n0)
    return (res - M) / np.sqrt(M)


def _valid(out):
    return bool(np.all(np.isfinite(np.asarray(out.position, float))))


class V1RComponents:
    def __init__(self):
        self.paths = PathInitEstimator(str(V1R / "path_init" / "model.pt"), sage_iterations=1)
        self.geom = GeomInitializer(str(V1R / "geom_init" / "model.pt"))
        self.router = LearnedRouter(str(V1R / "router" / "model.pt"))


def tier1(comp, y, context):
    t0 = time.perf_counter()
    x = network_input(y, context)
    cond, _ = comp.router.route(x)
    est, _ = fast_channel_estimate(comp.paths, x, y, context)
    out = branch_localize(est, y, context, cond, comp.geom)
    return out, est, cond, time.perf_counter() - t0


def tier2(comp, y, context, est1, cond):
    t0 = time.perf_counter()
    cfg = context.config
    est = sage(y, context, est1, max_iter=2, tol=cfg.estimator.sage_tol).estimate if est1.num_paths else est1
    if cond == "olos" and est.num_paths >= 2:
        e = est.sorted_by_delay()
        cands = [branch_localize(est, y, context, "olos", comp.geom)]
        try:
            c2 = dataclasses.replace(context, config=cfg.replace(**{"estimator.olos_lma_top_k": 2}))
            r = olos_localize(e, c2, cfg.estimator.exip_weighting, cfg.estimator.olos_alpha_step,
                              cfg.estimator.olos_alpha_max, pairs=cfg.estimator.olos_pairs, top_k=2)
            cands.append(PositionEstimate(r.position, r.orientation, "olos", r.scatterers, e, r.cost))
        except (ValueError, np.linalg.LinAlgError):
            pass
        out = min(cands, key=lambda o: o.cost if np.isfinite(o.cost) else np.inf)
    else:
        out = localize(est, context, cond)
    return out, est, time.perf_counter() - t0


def tier3(y, context_strong):
    t0 = time.perf_counter()
    out = ClassicalEstimator("unknown").estimate(y, context_strong)
    return out, time.perf_counter() - t0


class TieredEstimator(Estimator):
    """The deployable V2-C system (reads the calibrated policy)."""
    requires_oracle = False
    name = "proposed_v2"

    def __init__(self, policy: dict):
        self.comp = V1RComponents()
        self.th = {int(k): v for k, v in policy["thresholds"].items()}

    def _th(self, G):
        keys = sorted(self.th)
        below = [k for k in keys if k <= G]
        return self.th[below[-1] if below else keys[0]]

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        """`context` must carry the strong v2.2 estimator settings (tier 3 and the classical sub-steps use them)."""
        th = self._th(y.shape[1])
        out1, est1, cond, t1 = tier1(self.comp, y, context)
        if _valid(out1) and _z(out1, est1, y, context) <= th["z1"]:
            out1.info["tier"] = 1
            return out1
        out2, est2, t2 = tier2(self.comp, y, context, est1, cond)
        if _valid(out2) and _z(out2, est2, y, context) <= th["z2"]:
            out2.info["tier"] = 2
            return out2
        out3, _ = tier3(y, context)
        out3.info["tier"] = 3
        return out3

"""Proposed pipeline (steps 6A + 6B): learned path init -> SAGE (few iterations) -> path validation (41)
-> LOS closed form / NLOS EXIP (classical single-start init) / OLOS EXIP from the learned geometric init (1 LMA)."""
from __future__ import annotations

import time

import numpy as np

from ..localization.exip_lma import exip_lma, pack, weight_matrix
from ..localization.pipeline import localize
from ..signal_processing.model import ObservationModel
from .base import Estimator, PositionEstimate
from .geom_init import GeomInitializer
from .path_init import PathInitEstimator


class ProposedEstimator(Estimator):
    requires_oracle = False

    def __init__(self, path_checkpoint: str, geom_checkpoint: str, sage_iterations: int = 1, condition: str = "auto"):
        self.paths = PathInitEstimator(path_checkpoint, sage_iterations=sage_iterations, condition=condition)
        self.geom = GeomInitializer(geom_checkpoint)
        self.condition = condition
        self.name = f"proposed_sage{sage_iterations}"

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        cfg = context.config
        est, info = self.paths.channel_estimate(y, context)
        cond = cfg.scenario.condition if self.condition == "auto" else self.condition
        t0 = time.perf_counter()
        if cond == "olos" and est.num_paths >= 2:
            e = est.sorted_by_delay()
            c = cfg.physics.speed_of_light
            tg = time.perf_counter()
            p0, a0, s0 = self.geom.predict(e, c)
            info["stages"]["geom_net"] = time.perf_counter() - tg
            tg = time.perf_counter()
            r = exip_lma(e, weight_matrix(e, context, cfg.estimator.exip_weighting), pack(p0, a0, s0, e.gain_tilde, False),
                         False, context.bs_position, c)
            info["stages"]["lma"] = time.perf_counter() - tg
            out = PositionEstimate(r.position, r.orientation, "olos", r.scatterers, e, r.cost,
                                   info={"lma_nfev": r.nfev, "lma_success": r.success})
        else:
            out = localize(est, context, cond)
        info["timing"]["localize"] = time.perf_counter() - t0
        # 6C gate scores (existing physical quantities): normalised residual likelihood and EXIP cost
        t0 = time.perf_counter()
        ch = out.channel if out.channel is not None else est
        M = y.size
        res = float(np.sum(np.abs(y - ObservationModel(context).mean(ch)) ** 2) / context.n0)
        info["gate"] = {"r_norm": res / M, "z": (res - M) / np.sqrt(M),
                        "v": float(out.cost) if np.isfinite(out.cost) else 0.0}
        info["timing"]["gate"] = time.perf_counter() - t0
        info["stages"]["gate"] = info["timing"]["gate"]
        info["stages"].setdefault("localize_other", info["timing"]["localize"] - info["stages"].get("geom_net", 0.0)
                                  - info["stages"].get("lma", 0.0))
        out.info.update(info)
        return out

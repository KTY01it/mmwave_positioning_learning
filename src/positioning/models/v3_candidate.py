"""V3 candidate (B2f3), the deployable estimator: condition unknown, no gate, no fallback.

  v1R learned path heatmap -> adaptive model-order SAGE (forward/backward GLRT, Pinto & Juntti TWC 2026, adapted)
  with local 4x off-grid residual search (Shahmansoori TWC 2018 components) and branch folding (F27), 3 SAGE
  iterations per middle loop -> condition by the classical Sec. IV-D chi-square test on the fits -> LOS closed form /
  NLOS EXIP / OLOS v1R learned geometry init + 1 LMA.
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import Estimator, PositionEstimate
from .geom_init import GeomInitializer
from .path_init import PathInitEstimator, network_input
from .router import branch_localize, classical_route
from .v3_local import aosage3

ROOT = Path(__file__).resolve().parents[3]
V1R = ROOT / "outputs" / "models" / "v1r"


class V3Candidate(Estimator):
    requires_oracle = False
    name = "v3_candidate"

    def __init__(self, it_mid: int = 3):
        self.pe = PathInitEstimator(str(V1R / "path_init" / "model.pt"), sage_iterations=1)
        self.geom = GeomInitializer(str(V1R / "geom_init" / "model.pt"))
        self.it_mid = it_mid

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        """`context` must carry the strong v2.2 estimator settings (EXIP weighting, branches, chi-square level)."""
        t0 = time.perf_counter()
        x = network_input(y, context)
        est = aosage3(self.pe, y, context, variant="B2f", x=x, it_mid=self.it_mid)
        t1 = time.perf_counter()
        cond, info = classical_route(est, context)
        out = branch_localize(est, y, context, cond, self.geom)
        out.info.update({"c_hat": cond, "paths": est.info.get("v3local", {}),
                         "timing": {"paths": t1 - t0, "localize": time.perf_counter() - t1}})
        return out

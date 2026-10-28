"""Benchmark v2 helpers (docs/benchmark_v2_protocol.md): the 'fast_routed' ablation (P8 learned router + fast path,
no gate) built from the frozen P8 / P6 components without modifying them."""
from __future__ import annotations

import time

from .base import Estimator, PositionEstimate
from .path_init import network_input
from .router import RoutedGatedEstimator, branch_localize, fast_channel_estimate


class FastRoutedEstimator(Estimator):
    requires_oracle = False
    name = "fast_routed"

    def __init__(self, path_ckpt: str, geom_ckpt: str, router_ckpt: str):
        self.r = RoutedGatedEstimator(path_ckpt, geom_ckpt, router="learned", router_ckpt=router_ckpt)

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        x = network_input(y, context)
        est, stages = fast_channel_estimate(self.r.paths, x, y, context)
        cond, probs = self.r.learned.route(x)
        out = branch_localize(est, y, context, cond, self.r.geom)
        out.info["routed"] = {"c_hat": cond, "t_fast": time.perf_counter() - t0, "stages": stages}
        return out

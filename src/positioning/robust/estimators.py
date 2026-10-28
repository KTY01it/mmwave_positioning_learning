"""Estimators of the robustness completion program. The frozen V3Candidate is used as it is; these classes only decide
what noise variance it receives and, for G <= g_low, which initial support the copied loop starts from."""
from __future__ import annotations

import dataclasses
import time

from ..localization.pipeline import ClassicalEstimator
from ..models.base import Estimator, PositionEstimate
from ..models.v3_candidate import V3Candidate
from . import loop as LP
from . import lowg as LG
from .noise import n0_floor


class Practical(Estimator):
    """B2f3 with an observation-only noise variance (noise = "E1" | "E2" | "E3"; None = use context.n0 as given) and an
    optional low-G rule (lowg in lowg.VARIANTS, applied when G <= g_low). For G > g_low, or without a rule, the frozen
    V3Candidate.estimate is called unchanged."""

    requires_oracle = False

    def __init__(self, noise: str | None = "E1", lowg: str | None = None, g_low: int = LG.G_LOW, it_mid: int = 3):
        self.frozen = V3Candidate(it_mid=it_mid)
        self.noise, self.lowg, self.g_low = noise, lowg, g_low
        self.name = "b2f3" + (f"+n0{noise}" if noise else "") + (f"+{lowg}" if lowg else "")

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        n0 = n0_floor(y, context, self.noise) if self.noise else float(context.n0)
        ctx = dataclasses.replace(context, n0=n0)
        if self.lowg is None or y.shape[1] > self.g_low:
            out = self.frozen.estimate(y, ctx)
        else:
            out = LP.run(self.frozen, y, ctx, **LG.VARIANTS[self.lowg])
        out.info["n0_used"] = n0
        out.info["t_total"] = time.perf_counter() - t0
        return out


def classical(sample, n0: float | None = None, single: bool = True, noise: str | None = None):
    """strong classical v2.2 (single=False) or classical_single on `sample` (which must carry the strong_classical
    profile), condition unknown, with the given noise variance (n0), an observation-only estimate (noise), or the
    context value. Returns (PositionEstimate, seconds)."""
    t0 = time.perf_counter()
    ctx = sample.context
    if noise is not None:
        n0 = n0_floor(sample.y, ctx, noise)
    if n0 is not None:
        ctx = dataclasses.replace(ctx, n0=float(n0))
    if single:
        ctx = dataclasses.replace(ctx, config=ctx.config.replace(**{"estimator.sage_max_iter": 1,
                                                                    "estimator.olos_lma_top_k": 1}))
    out = ClassicalEstimator("unknown").estimate(sample.y, ctx)
    return out, time.perf_counter() - t0

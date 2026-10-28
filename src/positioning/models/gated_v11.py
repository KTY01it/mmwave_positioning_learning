"""Gated proposed estimator v1.1 (user decision 2026-10-07, docs/gate_v1_1.md).

Identical to the frozen v1.0 (`gated.py`: same networks, same z / v_EXIP thresholds, same calibration, same strong v2.2
fallback) plus one validity check placed before the thresholds: a fast-path output whose position is not finite
(no / too few paths survived) is never accepted. v1.0 could accept it because the EXIP-cost score is set to 0 for a
failed fit (P14 interpretation, section 6). v1.0 is kept unchanged for the pre-registered results.
"""
from __future__ import annotations

import time

import numpy as np

from ..localization.pipeline import ClassicalEstimator
from .base import PositionEstimate
from .gated import GatedProposedEstimator, _strong_context

GATE_VERSION = "1.1"


def valid_fast_output(position) -> bool:
    return bool(np.all(np.isfinite(np.asarray(position, float))))


class GatedProposedEstimatorV11(GatedProposedEstimator):
    name = "gated_proposed_v1_1"

    def accept_v11(self, out: PositionEstimate, G: int) -> bool:
        return valid_fast_output(out.position) and self.accept(out.info["gate"], G)

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        out = self.fast.estimate(y, context)
        t_fast = time.perf_counter() - t0
        if self.accept_v11(out, y.shape[1]):
            out.info["gated"] = {"accepted": True, "t_fast": t_fast, "t_fallback": 0.0, "version": GATE_VERSION}
            return out
        t1 = time.perf_counter()
        fb = ClassicalEstimator(self.condition).estimate(y, _strong_context(context))
        fb.info["gated"] = {"accepted": False, "t_fast": t_fast, "t_fallback": time.perf_counter() - t1,
                            "fast_gate": out.info["gate"], "fast_valid": valid_fast_output(out.position),
                            "version": GATE_VERSION}
        fb.info.setdefault("timing", {})["gate_fast_path"] = t_fast
        return fb

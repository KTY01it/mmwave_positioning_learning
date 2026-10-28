"""Prior-aware wrapper for any estimator: if the estimate leaves the prior region (P3 v1 box, +margin) or has
|alpha| > alpha_max + margin, return the prior mean. This is the simplest way for a classical estimator to use the
same prior that a learned model absorbs from its training set; learned methods must beat THIS at low SNR."""
from __future__ import annotations

import numpy as np

from .base import Estimator, PositionEstimate


class PriorFallback(Estimator):
    def __init__(self, inner: Estimator, p_low=(2.0, 0.0), p_high=(4.0, 0.3), alpha_max=0.5, margin=0.5):
        self.inner = inner
        self.lo, self.hi = np.asarray(p_low, float), np.asarray(p_high, float)
        self.alpha_max, self.margin = alpha_max, margin
        self.name = f"{getattr(inner, 'name', 'estimator')}+prior_fallback"
        self.requires_oracle = inner.requires_oracle

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        out = self.inner.estimate(y, context, gt)
        p, a = out.position, out.orientation
        ok = np.all(np.isfinite(p)) and np.isfinite(a) and np.all(p >= self.lo - self.margin) \
            and np.all(p <= self.hi + self.margin) and abs(a) <= self.alpha_max + self.margin
        if not ok:
            out.position, out.orientation = (self.lo + self.hi) / 2, 0.0
            out.info["prior_fallback"] = True
        return out

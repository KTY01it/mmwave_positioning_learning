"""Implementation repair (not a scientific variant): BS-frame canonicalisation.

The frozen GeomInitNet learned start (models/geom_init.py::predict) returns positions in the training frame, which has the
BS at the origin, and ignores context.bs_position. Every benchmark and confirmation used that frame (BS at the origin), so
no reported number is affected; with a BS elsewhere the learned start would be offset and the estimate is then not
translation-equivariant (F1 test: up to 2.9 m for B2f3R-v2). The wrapper moves the problem into the BS-centred frame and the
result back. Angles are relative to the arrays and unchanged; the BS array axis defines the frame orientation."""
from __future__ import annotations

import copy
import dataclasses

import numpy as np


class BSFrame:
    def __init__(self, estimator):
        self.inner = estimator
        self.name = f"bsframe[{getattr(estimator, 'name', type(estimator).__name__)}]"
        self.requires_oracle = False

    def estimate(self, y, context, gt=None):
        b = np.asarray(context.bs_position, float)
        if not np.any(b):
            return self.inner.estimate(y, context)
        out = self.inner.estimate(y, dataclasses.replace(context, bs_position=np.zeros_like(b)))
        out = copy.copy(out)
        out.position = np.asarray(out.position, float) + b
        if getattr(out, "scatterers", None) is not None:
            out.scatterers = np.asarray(out.scatterers, float) + b
        return out

"""Common estimator interface shared by classical, AI and hybrid methods.

An estimator sees ONLY (y, context). Ground truth is passed exclusively to estimators that declare
``requires_oracle = True``; their outputs are tagged ``oracle=True`` so they can never be mixed up
with non-oracle results in tables.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PositionEstimate:
    position: np.ndarray
    orientation: float
    condition: str | None = None
    scatterers: np.ndarray | None = None
    channel: object = None  # ChannelEstimate (final) if the method produces one
    cost: float = float("nan")
    oracle: bool = False
    runtime_s: float = float("nan")
    info: dict = field(default_factory=dict)

    @classmethod
    def failed(cls, reason: str) -> "PositionEstimate":
        return cls(np.full(2, np.nan), float("nan"), info={"failed": reason})


class Estimator(ABC):
    name: str = "estimator"
    requires_oracle: bool = False

    @abstractmethod
    def estimate(self, y: np.ndarray, context, gt=None) -> PositionEstimate:
        """``gt`` is None unless requires_oracle."""

    def __call__(self, sample) -> PositionEstimate:
        t = time.perf_counter()
        out = self.estimate(sample.y, sample.context, sample.gt if self.requires_oracle else None)
        out.runtime_s = time.perf_counter() - t
        out.oracle = self.requires_oracle
        return out

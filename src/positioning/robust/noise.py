"""R1: observation-only estimates of the noise variance and a wrapper that feeds them to an unchanged estimator.

Principle (standard noise-floor estimation): remove the known unit-modulus pilot phase, apply a unitary receive-beamspace
transform and a unitary inverse DFT over the subcarriers. A cell that contains no path is exactly CN(0, N0); the channel
is sparse in (delay, AoA), so most cells are signal-free and a robust location of the cell powers estimates N0.
Nothing here reads the true N0, the ground truth or the propagation condition.

  E1  median of the cell powers / ln 2
  E2  as E1 after a Hann taper over receive elements and over subcarriers (unit mean square, so the noise variance of a
      cell is still N0); lower leakage of strong off-grid paths
  E3  lower quartile of the cell powers / (-ln 0.75)
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..models.base import Estimator, PositionEstimate
from ..signal_processing import beamspace

KINDS = ("E1", "E2", "E3")


def _taper(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return w / np.sqrt(np.mean(w ** 2))


def cell_powers(y: np.ndarray, context, taper: bool = False) -> np.ndarray:
    """|.|^2 of the [delay, beam, rx-beam] cells; signal-free cells are exponential with mean N0."""
    cfg = context.config
    Ur = beamspace.dictionary(cfg.array.num_rx, cfg.physics.wavelength_c, cfg.antenna_spacing, cfg.array.index_origin,
                              1, "centered", True)
    yd = y * np.conj(context.x[:, :, 0])[:, :, None]                     # pilots are unit-modulus
    if taper:
        yd = yd * _taper(y.shape[0])[:, None, None] * _taper(y.shape[2])[None, None, :]
    t = np.fft.ifft(yd @ np.conj(Ur), axis=0) * np.sqrt(y.shape[0])
    return np.abs(t).ravel() ** 2


def n0_floor(y: np.ndarray, context, kind: str = "E1") -> float:
    if kind == "E1":
        return float(np.median(cell_powers(y, context)) / np.log(2.0))
    if kind == "E2":
        return float(np.median(cell_powers(y, context, taper=True)) / np.log(2.0))
    if kind == "E3":
        return float(np.quantile(cell_powers(y, context), 0.25) / -np.log(0.75))
    raise ValueError(kind)


def with_n0(sample, n0: float):
    """The same sample with context.n0 replaced (the estimator then never sees the simulator's N0)."""
    return dataclasses.replace(sample, context=dataclasses.replace(sample.context, n0=float(n0)))


class EstimatedNoise(Estimator):
    """inner estimator run with N0 := noise-floor estimate computed from (y, pilots) only."""

    requires_oracle = False

    def __init__(self, inner: Estimator, kind: str = "E1"):
        self.inner, self.kind = inner, kind
        self.name = f"{getattr(inner, 'name', 'estimator')}+n0{kind}"

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        n0 = n0_floor(y, context, self.kind)
        out = self.inner.estimate(y, dataclasses.replace(context, n0=n0))
        out.info["n0_hat"] = n0
        return out

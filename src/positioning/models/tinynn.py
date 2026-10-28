"""Benchmark v2 baseline: algorithm-supervised tiny NN of
A. Shastri, S. Blandino, C. Gentile, C. Lai, P. Casari, "Algorithm-Supervised Millimeter Wave Indoor Localization
Using Tiny Neural Networks", IEEE TWC 24(4):3177-3191, 2025 (DOI 10.1109/TWC.2025.3528632).
Protocol: docs/benchmark_v2_protocol.md.

Verbatim: ADoA construction (reference = minimum delay; remaining AoAs sorted ascending, minus the reference AoA,
wrapped to [-pi, pi), radians), algorithm labels (not ground truth), 5 % label-outlier removal, MLP
(ceil(kappa Ni), same, ceil(./2), 2) with ReLU everywhere incl. the output (Eq. 3), loss Eq. (4), Adam, the Table II
hyper-parameter grid, one network per scenario (here: per G), position-only output.
ADAPTATIONS: supervising algorithm = strong classical v2.2 (unknown condition) instead of JADE, on its own
DCS-SOMP + SAGE path estimates (no DBSCAN: specular simulated paths); input extended with the AoD and c*tau of the same
paths (single BS + random scatterers: ADoA alone does not identify position); K_max = 4 with zero padding.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .base import Estimator, PositionEstimate

K_MAX = 4
N_IN = 3 * K_MAX - 1


def tinynn_features(delay, aod, aoa, gain, c: float) -> np.ndarray:
    """[ADoA (K_MAX-1), AoD (K_MAX), c*tau (K_MAX)] with zero padding; reference path first, others sorted by AoA."""
    f = np.zeros(N_IN, np.float32)
    K = len(delay)
    if K == 0:
        return f
    ref = int(np.argmin(delay))
    others = [k for k in range(K) if k != ref]
    others = sorted(others, key=lambda k: -abs(gain[k]))[:K_MAX - 1]     # keep the strongest others [DESIGN: K > K_MAX]
    others = sorted(others, key=lambda k: aoa[k])                        # ascending AoA (paper)
    adoa = [np.angle(np.exp(1j * (aoa[k] - aoa[ref]))) for k in others]  # wrap to [-pi, pi)
    order = [ref] + others
    f[:len(adoa)] = adoa
    f[K_MAX - 1:K_MAX - 1 + len(order)] = [aod[k] for k in order]
    f[2 * K_MAX - 1:2 * K_MAX - 1 + len(order)] = [c * delay[k] for k in order]
    return f


class TinyNN(nn.Module):
    def __init__(self, n_in: int = N_IN, kappa: float = 0.9, dropout: float = 0.2):
        super().__init__()
        h1 = int(np.ceil(kappa * n_in))
        h3 = int(np.ceil(h1 / 2))
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(h1, h1), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(h1, h3), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(h3, 2), nn.ReLU())                     # Eq. (3): ReLU at the output too
        self.sizes = (n_in, h1, h1, h3, 2)

    def forward(self, x):
        return self.net(x)


class TinyNNEstimator(Estimator):
    """Strong v2.2 DCS-SOMP + SAGE channel estimate -> features -> per-G tiny NN (largest trained G <= G)."""

    requires_oracle = False
    name = "tiny_nn"

    def __init__(self, checkpoint: str):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.nets = {}
        for G, d in ck["per_G"].items():
            net = TinyNN(**d["model_kwargs"])
            net.load_state_dict(d["state_dict"])
            self.nets[int(G)] = net.eval()

    def net_for(self, G: int):
        keys = sorted(self.nets)
        below = [k for k in keys if k <= G]
        return self.nets[below[-1] if below else keys[0]]

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        from ..localization.pipeline import ClassicalEstimator
        t0 = time.perf_counter()
        est, info = ClassicalEstimator("unknown").channel_estimate(y, context)   # caller passes the strong context
        t1 = time.perf_counter()
        if est.num_paths == 0:
            return PositionEstimate.failed("no path detected")
        x = tinynn_features(est.delay, est.aod, est.aoa, est.gain_tilde, context.config.physics.speed_of_light)
        with torch.no_grad():
            p = self.net_for(y.shape[1])(torch.from_numpy(x)[None])[0].numpy().astype(float)
        return PositionEstimate(p, float("nan"), None, channel=est,
                                info={"timing": {"coarse": info["timing"]["coarse"], "sage": info["timing"]["sage"],
                                                 "net": time.perf_counter() - t1, "channel": t1 - t0}})

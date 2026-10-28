"""Benchmark v2 baseline: architecture-faithful adaptation of the 3D CNN of
C. Wu et al., "Learning to Localize: A 3D CNN Approach to User Positioning in Massive MIMO-OFDM Systems",
IEEE TWC 20(7):4556-4570, 2021 (DOI 10.1109/TWC.2021.3060482). Spec: docs/benchmark_v2_protocol.md.

Verbatim from the paper: CNA layers (conv, no bias, 'same' padding -> BN -> ReLU, Eqs. 32-33), the two asymmetric
feature-refinement branches (Fig. 4(a)), three 3D Inception modules n = 1, 2, 4 with max-pools after n = 1 and n = 4
(Fig. 3, Fig. 4(b), pool projection [3 3 3] as printed), global average pooling, linear output (Eq. 30), loss Eq. (31)
(squared Euclidean error in metres + L2 weight decay).
ADAPTATIONS: input = receiver-side angle-delay power map from G random Tx beams (ULA-ULA) instead of the full-CSI UPA
ADCPM; noise floor known (= 1 after normalisation) instead of Eq. (25); per-sample Frobenius normalisation (45 dB SNR
range); output (x, y). The paper estimates no orientation: this baseline returns position only.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import Estimator, PositionEstimate
from .features import P_CENTER, P_SCALE, correlation_cube


# ---------------------------------------------------------------------------------------------------------- input
def wu_power_map(y: np.ndarray, context) -> np.ndarray:
    """float32 [AoD 65, AoA 65, delay N]: |unitary IFFT_n of the correlation cube|^2 (noise-only entries ~ Exp(1))."""
    c = correlation_cube(y, context)                   # [N, Nt, Nr], noise CN(0, 1)
    N = c.shape[0]
    t = np.fft.ifft(c, axis=0) * np.sqrt(N)
    return np.transpose(np.abs(t) ** 2, (1, 2, 0)).astype(np.float32)


def wu_preprocess(P: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq. (27) noise reduction with xi = beta * (known floor 1), then per-sample Frobenius normalisation.
    P: [B, 65, 65, N] -> [B, 1, 65, 65, N]."""
    X = torch.where(P > beta, P, torch.zeros_like(P))
    n = torch.linalg.vector_norm(X.flatten(1), dim=1).clamp_min(1e-12)
    return (X / n[:, None, None, None])[:, None]


# ---------------------------------------------------------------------------------------------------------- network
class CNA(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, k, stride=1, padding="same", bias=False)
        self.bn = nn.BatchNorm3d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MaxPoolSame(nn.Module):
    """TF-style 'SAME' max pooling (out = ceil(in / stride)); reproduces every shape printed in Fig. 3."""

    def __init__(self, k, s):
        super().__init__()
        self.k, self.s = tuple(k), tuple(s)

    def forward(self, x):
        pads = []
        for d in range(3)[::-1]:
            n = x.shape[2 + d]
            out = math.ceil(n / self.s[d])
            tot = max((out - 1) * self.s[d] + self.k[d] - n, 0)
            pads += [tot // 2, tot - tot // 2]
        return F.max_pool3d(F.pad(x, pads, value=float("-inf")), self.k, self.s)


class AvgPoolSame3(nn.Module):
    """3x3x3 / stride-1 'SAME' average pooling, padding excluded from the mean."""

    def forward(self, x):
        xp = F.pad(x, [1] * 6)
        ones = F.pad(torch.ones_like(x[:, :1]), [1] * 6)
        return F.avg_pool3d(xp, 3, 1) / F.avg_pool3d(ones, 3, 1)


class Refinement(nn.Module):
    def __init__(self, cin=1):
        super().__init__()
        self.left = nn.Sequential(CNA(cin, 16, (1, 5, 7)), MaxPoolSame((1, 3, 5), (1, 2, 2)), CNA(16, 32, (1, 3, 5)))
        self.right = nn.Sequential(CNA(cin, 16, (3, 1, 7)), MaxPoolSame((1, 3, 5), (1, 2, 2)), CNA(16, 32, (3, 1, 5)))
        self.fuse = nn.Sequential(CNA(64, 128, (3, 3, 3)), MaxPoolSame((3, 3, 5), (2, 2, 2)))

    def forward(self, x):
        return self.fuse(torch.cat([self.left(x), self.right(x)], 1))


class Inception3D(nn.Module):
    def __init__(self, cin, n, pool_proj_kernel=3):
        super().__init__()
        self.b1 = CNA(cin, 64 * n, 1)
        self.b2 = nn.Sequential(CNA(cin, 48 * n, 1), CNA(48 * n, 64 * n, 3))
        self.b3 = nn.Sequential(CNA(cin, 64 * n, 1), CNA(64 * n, 96 * n, 3), CNA(96 * n, 96 * n, 3))
        self.b4 = nn.Sequential(AvgPoolSame3(), CNA(cin, 32 * n, pool_proj_kernel))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], 1)


class Wu3DCNN(nn.Module):
    def __init__(self, out_dim: int = 2, pool_proj_kernel: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            Refinement(1),
            Inception3D(128, 1, pool_proj_kernel), MaxPoolSame((2, 5, 5), (2, 2, 2)),
            Inception3D(256, 2, pool_proj_kernel),
            Inception3D(512, 4, pool_proj_kernel), MaxPoolSame((2, 3, 3), (2, 2, 2)),
        )
        self.fc = nn.Linear(1024, out_dim)
        for m in self.modules():                       # TF defaults (Glorot uniform) [DESIGN: init not specified]
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.body(x).mean(dim=(2, 3, 4)))


_PC = torch.tensor(P_CENTER, dtype=torch.float32)
_PS = torch.tensor(P_SCALE, dtype=torch.float32)


def wu_loss(pred_norm: torch.Tensor, pos_m: torch.Tensor) -> torch.Tensor:
    """Eq. (31) data term: squared Euclidean error in metres (network predicts normalised coordinates)."""
    p = pred_norm * _PS.to(pred_norm) + _PC.to(pred_norm)
    return ((p - pos_m) ** 2).sum(1).mean()


# ---------------------------------------------------------------------------------------------------------- estimator
class WuEstimator(Estimator):
    requires_oracle = False
    name = "wu_3dcnn"

    def __init__(self, checkpoint: str, device: str = "cpu", threads: int = 1):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.net = Wu3DCNN(**ck["model_kwargs"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval().to(device)
        self.beta = float(ck["beta"])
        self.device = device
        if device == "cpu":
            torch.set_num_threads(threads)

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        P = torch.from_numpy(wu_power_map(y, context))[None].to(self.device)
        t1 = time.perf_counter()
        with torch.no_grad():
            v = self.net(wu_preprocess(P, self.beta)).float().cpu().numpy()[0]
        p = v * P_SCALE + P_CENTER
        return PositionEstimate(p, float("nan"), None,
                                info={"timing": {"features": t1 - t0, "net": time.perf_counter() - t1}})

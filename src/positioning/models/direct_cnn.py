"""P4 baseline A: direct regression y -> (p, alpha) with a CNN on the angle-angle-delay image."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .base import Estimator, PositionEstimate
from .features import angle_delay_image, decode_pose


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DirectCNN(nn.Module):
    def __init__(self, in_ch: int = 20, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(in_ch, w), nn.MaxPool2d(2),            # 65 -> 32
            _block(w, 2 * w), _block(2 * w, 2 * w), nn.MaxPool2d(2),   # -> 16
            _block(2 * w, 4 * w), _block(4 * w, 4 * w), nn.MaxPool2d(2),  # -> 8
            _block(4 * w, 8 * w), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(8 * w + 1, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))

    def forward(self, x, g):
        return self.head(torch.cat([self.features(x).flatten(1), g], dim=1))


def pose_loss(pred, target):
    """Huber on the normalised position + (1 - cos) on the orientation (from the unit-normalised (cos, sin))."""
    lp = nn.functional.smooth_l1_loss(pred[:, :2], target[:, :2])
    c = nn.functional.normalize(pred[:, 2:], dim=1)
    la = (1 - (c * target[:, 2:]).sum(1)).mean()
    return lp + la


class DirectCNNEstimator(Estimator):
    name = "direct_cnn"
    requires_oracle = False

    def __init__(self, checkpoint: str, threads: int = 1):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = DirectCNN(**ck["model_kwargs"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        torch.set_num_threads(threads)

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        x = torch.from_numpy(angle_delay_image(y, context))[None]
        g = torch.tensor([[np.log2(y.shape[1] / 32.0)]], dtype=torch.float32)
        t1 = time.perf_counter()
        with torch.no_grad():
            v = self.model(x, g)[0].numpy()
        p, a = decode_pose(v)
        return PositionEstimate(p, a, info={"timing": {"coarse": t1 - t0, "sage": 0.0,
                                                        "localize": time.perf_counter() - t1}})

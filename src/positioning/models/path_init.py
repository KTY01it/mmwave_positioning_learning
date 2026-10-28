"""Step 6A: learned path initializer (replaces the coarse stage only) -> SAGE (budget b) -> LOS / EXIP-LMA.

Network: small fully convolutional U-Net on the angle-angle-delay image (features.angle_delay_image, unitary 1x grid,
Nt x Nr = 65 x 65) plus a constant log2(G/32) channel. Output per grid cell (CenterNet-style):
  0  path objectness logit (target: Gaussian bump, sigma = 1 cell, around the cell of every true path)
  1  sub-bin offset of the AoD index   in [-0.5, 0.5]     (supervised at path cells only)
  2  sub-bin offset of the AoA index   in [-0.5, 0.5]
  3  delay tau / Ts
Decoding: 3x3 local maxima of sigmoid(objectness) above a threshold calibrated on the validation split (frozen before
test), at least one path, at most max_paths. Gains: joint least squares given the decoded (tau, AoD, AoA). Then the
frozen SAGE and localisation modules.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ..signal_processing.beamspace import sine_to_angle
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage
from .base import Estimator, PositionEstimate
from .features import angle_delay_image


# ---------------------------------------------------------------------------------------------------------------------
# grid <-> angles (unitary 1x grid of features.py: sin(theta_m) = 2 (m - (N-1)/2) / N, m = 0..N-1, d = lambda/2)
# ---------------------------------------------------------------------------------------------------------------------
def sine_to_index(sine, n):
    return np.asarray(sine) * n / 2 + (n - 1) / 2


def index_to_sine(idx, n):
    return (np.asarray(idx, float) - (n - 1) / 2) * 2 / n


# ---------------------------------------------------------------------------------------------------------------------
def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PathInitNet(nn.Module):
    def __init__(self, in_ch: int = 21, width: int = 16):
        super().__init__()
        w = width
        self.e1 = nn.Sequential(_conv(in_ch, w), _conv(w, w))                      # 65
        self.e2 = nn.Sequential(nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                _conv(2 * w, 2 * w))                               # 33
        self.e3 = nn.Sequential(nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                _conv(4 * w, 4 * w), _conv(4 * w, 4 * w))           # 17
        self.d2 = _conv(4 * w + 2 * w, 2 * w)
        self.d1 = _conv(2 * w + w, w)
        self.head = nn.Conv2d(w, 4, 1)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[0] = -4.6  # prior objectness ~ 1 %

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([nn.functional.interpolate(e3, size=e2.shape[-2:], mode="bilinear",
                                                          align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([nn.functional.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                                                          align_corners=False), e1], 1))
        return self.head(d1)


def network_input(y, context) -> np.ndarray:
    img = angle_delay_image(y, context)                                           # [D, Nt, Nr]
    g = np.full((1,) + img.shape[1:], np.log2(y.shape[1] / 32.0), dtype=np.float32)
    return np.concatenate([img, g], axis=0)


def make_targets(paths, cfg, ts: float):
    """Heatmap [Nt, Nr], regression maps [3, Nt, Nr] and a mask [Nt, Nr] of path cells."""
    nt, nr = cfg.array.num_tx, cfg.array.num_rx
    heat = np.zeros((nt, nr), np.float32)
    reg = np.zeros((3, nt, nr), np.float32)
    mask = np.zeros((nt, nr), np.float32)
    ii, jj = np.meshgrid(np.arange(nt), np.arange(nr), indexing="ij")
    order = np.argsort(-np.abs(paths.gain_tilde))  # stronger path wins a shared cell
    for k in order[::-1]:
        ct = sine_to_index(np.sin(paths.aod[k]), nt)
        cr = sine_to_index(np.sin(paths.aoa[k]), nr)
        it, ir = int(np.clip(np.round(ct), 0, nt - 1)), int(np.clip(np.round(cr), 0, nr - 1))
        heat = np.maximum(heat, np.exp(-((ii - it) ** 2 + (jj - ir) ** 2) / 2.0).astype(np.float32))
        reg[:, it, ir] = (ct - it, cr - ir, paths.delay[k] / ts)
        mask[it, ir] = 1.0
    return heat, reg, mask


def path_loss(out, heat, reg, mask, alpha=2.0, beta=4.0):
    """CenterNet focal loss on the objectness + L1 on offsets/delay at path cells."""
    p = torch.sigmoid(out[:, 0]).clamp(1e-6, 1 - 1e-6)
    pos = (heat >= 0.999).float()
    lpos = -((1 - p) ** alpha * torch.log(p) * pos).sum()
    lneg = -((1 - heat) ** beta * p ** alpha * torch.log(1 - p) * (1 - pos)).sum()
    npos = pos.sum().clamp(min=1.0)
    m = mask[:, None]
    lreg = (torch.abs(out[:, 1:] - reg) * m).sum() / (m.sum() * 3).clamp(min=1.0)
    return (lpos + lneg) / npos + lreg


def decode(out: np.ndarray, cfg, threshold: float, max_paths: int = 4):
    """out [4, Nt, Nr] -> list of (prob, aod, aoa, tau/Ts)."""
    prob = 1 / (1 + np.exp(-out[0]))
    P = np.pad(prob, 1, constant_values=-1)
    loc = np.ones_like(prob, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= prob >= P[1 + di:1 + di + prob.shape[0], 1 + dj:1 + dj + prob.shape[1]]
    cells = sorted(zip(*np.nonzero(loc)), key=lambda c: -prob[c])
    keep = [c for c in cells if prob[c] >= threshold][:max_paths] or cells[:1]
    nt, nr = prob.shape
    e = cfg.estimator
    res = []
    for i, j in keep:
        st = float(np.clip(index_to_sine(i + np.clip(out[1, i, j], -0.5, 0.5), nt), -1, 1))
        sr = float(np.clip(index_to_sine(j + np.clip(out[2, i, j], -0.5, 0.5), nr), -1, 1))
        res.append((float(prob[i, j]), float(sine_to_angle(st, e.aod_branch)), float(sine_to_angle(sr, e.aoa_branch)),
                    float(max(out[3, i, j], 0.0))))
    return res


def ls_gains(model: ObservationModel, y, delay, aod, aoa) -> np.ndarray:
    B = np.stack([model.path_basis(t, a, b).ravel() for t, a, b in zip(delay, aod, aoa)], axis=1)
    return np.linalg.lstsq(B, y.ravel(), rcond=None)[0]


def validate_paths(y, context, est: ChannelEstimate, min_paths: int = 1) -> ChannelEstimate:
    """Physics validation of learned paths with the frozen baseline's test (41): a path is kept only if removing it
    from the per-subcarrier least-squares fit (same atoms and statistic as DCS-SOMP) loses at least delta of energy.
    The weakest failing path is removed and the test repeated."""
    from ..signal_processing import beamspace
    from ..signal_processing.coarse import _atom, _project
    from ..signal_processing.dcs_somp import detection_threshold
    cfg = context.config
    e = cfg.estimator
    model = ObservationModel(context)
    yb = beamspace.stack_observation(y).astype(complex)
    cols = cfg.array.num_tx * cfg.array.num_rx * e.offgrid_threshold_factor
    delta = detection_threshold(context.n0, cfg.physics.num_subcarriers, cols, e.pfa)
    keep = list(range(est.num_paths))
    while len(keep) > min_paths:
        atoms = [_atom(model, est.aod[k], est.aoa[k]) for k in keep]
        r_all = np.sum(np.abs(_project(yb, atoms)[1]) ** 2)
        gains = [np.sum(np.abs(_project(yb, atoms[:i] + atoms[i + 1:])[1]) ** 2) - r_all for i in range(len(keep))]
        i = int(np.argmin(gains))
        if gains[i] >= delta:
            break
        keep.pop(i)
    return est.subset(keep)


class PathInitEstimator(Estimator):
    """Learned path initializer -> SAGE (sage_iterations; 0 = none) -> LOS / EXIP-LMA."""

    requires_oracle = False

    def __init__(self, checkpoint: str, sage_iterations: int | None = None, condition: str = "auto",
                 threshold: float | None = None, threads: int = 1, validate: bool = True):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.net = PathInitNet(**ck["model_kwargs"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.threshold = ck.get("threshold", 0.5) if threshold is None else threshold
        self.sage_iterations = sage_iterations
        self.condition = condition
        self.validate = validate  # physics validation of the learned paths with test (41)
        self.name = f"path_init_sage{'full' if sage_iterations is None else sage_iterations}"
        torch.set_num_threads(threads)

    def channel_estimate(self, y, context):
        cfg = context.config
        t0 = time.perf_counter()
        x = torch.from_numpy(network_input(y, context))[None]
        t_feat = time.perf_counter()
        with torch.no_grad():
            out = self.net(x)[0].numpy()
        t_net = time.perf_counter()
        dec = decode(out, cfg, self.threshold, cfg.estimator.max_paths)
        Ts = cfg.physics.sampling_period
        delay = np.array([d[3] * Ts for d in dec])
        aod = np.array([d[1] for d in dec])
        aoa = np.array([d[2] for d in dec])
        model = ObservationModel(context)
        gain = ls_gains(model, y, delay, aod, aoa)
        init = ChannelEstimate(delay, aod, aoa, gain, info={"probabilities": [d[0] for d in dec]})
        t1 = time.perf_counter()
        b = cfg.estimator.sage_max_iter if self.sage_iterations is None else self.sage_iterations
        if b > 0:
            sr = sage(y, context, init, max_iter=b, tol=cfg.estimator.sage_tol)
            est, info = sr.estimate, {"sage": sr}
        else:
            est, info = init, {}
        t2 = time.perf_counter()
        if self.validate and est.num_paths > 1:
            est = validate_paths(y, context, est, cfg.estimator.min_paths)
        t3 = time.perf_counter()
        info.update({"coarse": init, "timing": {"coarse": t1 - t0 + t3 - t2, "sage": t2 - t1},
                     "stages": {"features": t_feat - t0, "path_net": t_net - t_feat, "decode_ls": t1 - t_net,
                                "sage": t2 - t1, "validation": t3 - t2}})
        return est, info

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        from ..localization.pipeline import localize
        est, info = self.channel_estimate(y, context)
        cond = context.config.scenario.condition if self.condition == "auto" else self.condition
        t0 = time.perf_counter()
        out = localize(est, context, cond)
        info["timing"]["localize"] = time.perf_counter() - t0
        out.info.update(info)
        return out

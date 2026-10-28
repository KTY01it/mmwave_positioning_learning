"""Step 6B: learned geometric initialisation for OLOS EXIP (replaces the alpha grid + many LMA starts of Eqs. 64-65).

Input: the estimated paths (tau, theta_Tx, theta_Rx, |h~|), sorted by delay. Permutation-equivariant set network
(DeepSets): per-path encoder -> mean/max pooled context -> global head (p, cos alpha, sin alpha) and per-path head
(scatterer s_k of that path). Training data: geometry only, drawn from the P3 v1 prior with the frozen scenario /
geometry / channel modules (no observation simulated), path parameters perturbed with a random, log-uniform scale to
cover the SNR range, and occasionally one path dropped (missed detection).

Initialisations offered to the single LMA start:
  direct      (p_hat, alpha_hat, s_hat) from the network
  alpha65     alpha_hat only, then Eq. (65) for every path pair (best initial EXIP cost), as the classical init does
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..signal_processing.model import ChannelEstimate
from .features import P_CENTER, P_SCALE

S_CENTER = np.array([2.25, 1.15])
S_SCALE = np.array([1.25, 0.85])  # P3 v1 scatterer box [1, 3.5] x [0.3, 2]
N_FEAT = 6


def path_features(est: ChannelEstimate, c: float) -> np.ndarray:
    """[K, 6]: c*tau/5 m, sin/cos AoD, sin/cos AoA, relative path power in units of 20 dB."""
    g = np.abs(est.gain_tilde)
    rel = 20 * np.log10(np.maximum(g, 1e-300) / max(g.max(), 1e-300)) / 20.0
    return np.stack([c * est.delay / 5.0, np.sin(est.aod), np.cos(est.aod), np.sin(est.aoa), np.cos(est.aoa),
                     np.clip(rel, -3, 0)], axis=1).astype(np.float32)


class GeomInitNet(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        w = width
        self.enc = nn.Sequential(nn.Linear(N_FEAT, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w))
        self.ctx = nn.Sequential(nn.Linear(2 * w, w), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 4))
        self.per = nn.Sequential(nn.Linear(2 * w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 2))

    def forward(self, x, mask):
        """x [B, K, 6], mask [B, K] (1 = path present) -> pose [B, 4], scatterers [B, K, 2] (normalised)."""
        h = self.enc(x) * mask[..., None]
        n = mask.sum(1, keepdim=True).clamp(min=1)
        mean = h.sum(1) / n
        mx = (h - 1e4 * (1 - mask[..., None])).amax(1)
        c = self.ctx(torch.cat([mean, mx], -1))
        pose = self.glob(c)
        per = self.per(torch.cat([h, c[:, None].expand_as(h)], -1))
        return pose, per


def geom_loss(pose, per, p_t, a_t, s_t, mask):
    lp = nn.functional.smooth_l1_loss(pose[:, :2], p_t)
    u = nn.functional.normalize(pose[:, 2:], dim=1)
    la = (1 - (u[:, 0] * torch.cos(a_t) + u[:, 1] * torch.sin(a_t))).mean()
    ls = (nn.functional.smooth_l1_loss(per, s_t, reduction="none").sum(-1) * mask).sum() / mask.sum().clamp(min=1)
    return lp + 10 * la + ls


class GeomInitializer:
    def __init__(self, checkpoint: str, threads: int = 1):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.net = GeomInitNet(**ck["model_kwargs"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        torch.set_num_threads(threads)

    def predict(self, est: ChannelEstimate, c: float):
        """est sorted by delay -> (p, alpha, scatterers [K, 2]) in that path order."""
        x = torch.from_numpy(path_features(est, c))[None]
        m = torch.ones(1, est.num_paths)
        with torch.no_grad():
            pose, per = self.net(x, m)
        pose, per = pose[0].numpy(), per[0].numpy()
        return pose[:2] * P_SCALE + P_CENTER, float(np.arctan2(pose[3], pose[2])), per * S_SCALE + S_CENTER


# ---------------------------------------------------------------------------------------------------------------------
# training data (geometry only, frozen system modules)
# ---------------------------------------------------------------------------------------------------------------------
def sample_geometries(spec_path: str, split, seeds, K_max: int = 4):
    """Arrays for seeds of an OLOS split component: features of the TRUE paths (unperturbed), targets, mask."""
    from ..config import load_config
    from ..system import channel, geometry
    from ..system.scenario import sample_scenario
    comps = [c for c in split.components if c.overrides.get("scenario.condition") == "olos"]
    cfg = load_config(spec_path, **comps[0].overrides)
    c = cfg.physics.speed_of_light
    B = len(seeds)
    X = np.zeros((B, K_max, 6), np.float32)
    S = np.zeros((B, K_max, 2), np.float32)
    M = np.zeros((B, K_max), np.float32)
    P = np.zeros((B, 2), np.float32)
    A = np.zeros(B, np.float32)
    for b, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        sc = sample_scenario(cfg, rng)
        g = geometry.compute_path_geometry(sc.bs_position, sc.ue_position, sc.orientation, sc.scatterers, False, c)
        gain, rho, _ = channel.draw_path_gains(g, cfg, rng)
        o = np.argsort(g.delay)
        est = ChannelEstimate(g.delay[o], g.aod[o], g.aoa[o], gain[o])
        k = est.num_paths
        X[b, :k] = path_features(est, c)
        S[b, :k] = (sc.scatterers[o] - S_CENTER) / S_SCALE
        M[b, :k] = 1
        P[b] = (sc.ue_position - P_CENTER) / P_SCALE
        A[b] = sc.orientation
    return X, S, M, P, A


def perturb(X, M, rng, c=299_792_000.0):
    """Estimation noise with a per-sample log-uniform scale; one path dropped with probability 0.15."""
    B, K, _ = X.shape
    u = 10 ** rng.uniform(-3, 0, size=(B, 1))
    X = X.copy()
    dtau = rng.standard_normal((B, K)) * 0.3 * u / 5.0                      # up to 0.3 m of path length
    X[..., 0] += dtau.astype(np.float32)
    for j_sin, j_cos in ((1, 2), (3, 4)):
        th = np.arctan2(X[..., j_sin], X[..., j_cos]) + rng.standard_normal((B, K)) * 0.02 * u
        X[..., j_sin], X[..., j_cos] = np.sin(th), np.cos(th)
    X[..., 5] += (rng.standard_normal((B, K)) * 0.05 * u).astype(np.float32)
    M = M.copy()
    drop = rng.random(B) < 0.15
    for b in np.nonzero(drop)[0]:
        present = np.nonzero(M[b])[0]
        if present.size > 2:
            M[b, rng.choice(present)] = 0
    return X * M[..., None], M

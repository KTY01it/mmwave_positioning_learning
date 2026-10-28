"""P5 baseline B: model-driven unrolled sparse estimator for Eq. (36), y_check[n] = Omega[n] h_check[n] + noise.

Group (joint-sparse over subcarriers) ALISTA/LAMP-style unrolling with the sample's own operator:
    x <- eta_{theta_t}( x + gamma_t / L * Omega^H (y - Omega x) ),   eta = group soft threshold over n,
Omega[n] x[n] = Z[n] x[n] U_Rx^T (Kronecker structure, Eqs. 37-39), L = max_n ||Z[n]||_2^2. Learned parameters:
gamma_t, theta_t per layer (2 x layers scalars), so the network adapts to the per-sample random beams by construction.
The sparse estimate replaces the DCS-SOMP atom search: its group-energy peaks are proposed, in order, to the
off-grid detector, which keeps the calibrated acceptance test (41) (a plain peak picker accepts leakage sidelobes as
paths at high SNR). SAGE and EXIP/LMA are unchanged (frozen baseline modules).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ..signal_processing import beamspace, delay_estimation
from ..signal_processing.coarse import coarse_estimate
from ..signal_processing.model import ChannelEstimate
from ..signal_processing.sage import sage
from .base import Estimator, PositionEstimate


def unitary_dictionaries(cfg):
    lam, d, o = cfg.physics.wavelength_c, cfg.antenna_spacing, cfg.array.index_origin
    Ut = beamspace.dictionary(cfg.array.num_tx, lam, d, o, 1, "centered", True)
    Ur = beamspace.dictionary(cfg.array.num_rx, lam, d, o, 1, "centered", True)
    st = beamspace.grid_sine(cfg.array.num_tx, lam, d, 1, "centered")
    sr = beamspace.grid_sine(cfg.array.num_rx, lam, d, 1, "centered")
    return Ut, Ur, st, sr


def operator_inputs(y, context):
    """Normalised observation and operator: y/sqrt(N0) [N,G,Nr], Z [N,G,Nbt], U_Rx [Nr,Nbr]."""
    Ut, Ur, _, _ = unitary_dictionaries(context.config)
    Z = beamspace.beam_projections(context.s, Ut)
    return (y / np.sqrt(context.n0)).astype(np.complex64), Z.astype(np.complex64), Ur.astype(np.complex64)


class GroupALISTA(nn.Module):
    def __init__(self, layers: int = 12):
        super().__init__()
        self.layers = layers
        self.log_gamma = nn.Parameter(torch.zeros(layers))           # step = exp(.)/L
        self.theta = nn.Parameter(torch.full((layers,), 2.0))        # threshold in noise-std units of the gradient

    @staticmethod
    def A(x, Z, Ur):          # x [B,N,Nbt,Nbr] -> [B,N,G,Nr]
        return Z @ x @ Ur.transpose(-1, -2)

    @staticmethod
    def AH(r, Z, Ur):         # r [B,N,G,Nr] -> [B,N,Nbt,Nbr]
        return Z.conj().transpose(-1, -2) @ r @ Ur.conj()

    def forward(self, y, Z, Ur, return_all=False):
        B, N = y.shape[:2]
        L = torch.linalg.matrix_norm(Z, ord=2).amax(dim=1) ** 2  # [B]
        L = L[:, None, None, None]
        # noise std of Omega^H n per entry ~ column norm of Z (U_Rx unitary); use the RMS column norm
        col = torch.sqrt((Z.abs() ** 2).sum(dim=2).mean(dim=(1, 2)))[:, None, None]  # [B,1,1]
        x = torch.zeros(B, N, Z.shape[-1], Ur.shape[-1], dtype=y.dtype)
        outs = []
        for t in range(self.layers):
            x = x + (torch.exp(self.log_gamma[t]) / L) * self.AH(y - self.A(x, Z, Ur), Z, Ur)
            norm = torch.sqrt((x.abs() ** 2).sum(dim=1) + 1e-12)                     # [B,Nbt,Nbr]
            thr = torch.relu(self.theta[t]) * torch.exp(self.log_gamma[t]) * col * np.sqrt(N) / L[:, 0]
            x = x * torch.relu(1 - thr / norm)[:, None]
            outs.append(x)
        return outs if return_all else x


def nmse_loss(xs, target):
    """Mean over layers (deep supervision, later layers weighted more) of ||x - h||^2 / ||h||^2."""
    den = (target.abs() ** 2).sum(dim=(1, 2, 3)) + 1e-12
    w = torch.linspace(0.2, 1.0, len(xs))
    return sum(wi * ((x - target).abs() ** 2).sum(dim=(1, 2, 3)).div(den).mean() for wi, x in zip(w, xs)) / w.sum()


def peak_proposals(x: np.ndarray, cfg, max_candidates: int = 8, rel_floor_db: float = -30.0):
    """Local maxima of the group energy of the unrolled estimate, strongest first, as (aod, aoa) proposals."""
    e = cfg.estimator
    _, _, st, sr = unitary_dictionaries(cfg)
    E = (np.abs(x) ** 2).sum(axis=0)
    if E.max() <= 0:
        return []
    P = np.pad(E, 1)
    loc = np.ones_like(E, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= E >= P[1 + di:1 + di + E.shape[0], 1 + dj:1 + dj + E.shape[1]]
    cand = sorted(zip(*np.nonzero(loc & (E >= E.max() * 10 ** (rel_floor_db / 10)))), key=lambda ij: -E[ij])
    return [(float(beamspace.sine_to_angle(st[i], e.aod_branch)), float(beamspace.sine_to_angle(sr[j], e.aoa_branch)))
            for i, j in cand[:max_candidates]]


def decode_paths(x: np.ndarray, cfg, max_paths: int = 4, rel_floor_db: float = -25.0) -> ChannelEstimate:
    """Peaks of the group energy -> paths. Sidelobes are suppressed by requiring a 3x3 local maximum and >= 2 bins
    from any stronger kept peak; angles refined by quadratic interpolation; delay/gain from (53)-(54)."""
    e = cfg.estimator
    _, _, st, sr = unitary_dictionaries(cfg)
    E = (np.abs(x) ** 2).sum(axis=0)                 # [Nbt, Nbr]
    if E.max() <= 0:
        return ChannelEstimate(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, complex))
    P = np.pad(E, 1)
    loc = np.ones_like(E, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= E >= P[1 + di:1 + di + E.shape[0], 1 + dj:1 + dj + E.shape[1]]
    cand = sorted(zip(*np.nonzero(loc & (E >= E.max() * 10 ** (rel_floor_db / 10)))), key=lambda ij: -E[ij])
    kept = []
    for i, j in cand:
        if all(max(abs(i - a), abs(j - b)) >= 2 for a, b in kept):
            kept.append((i, j))
        if len(kept) >= max_paths:
            break

    def interp(v, k, grid):
        if 0 < k < len(v) - 1:
            a, b, c = v[k - 1], v[k], v[k + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den != 0 else 0.0
            off = float(np.clip(off, -0.5, 0.5))
            return grid[k] + off * (grid[1] - grid[0])
        return grid[k]
    Ts = cfg.physics.sampling_period
    out = {"delay": [], "aod": [], "aoa": [], "gain": []}
    for i, j in kept:
        s_t = interp(E[:, j], i, st)
        s_r = interp(E[i, :], j, sr)
        coef = x[:, i, j]
        tau = delay_estimation.periodogram_delay(coef, Ts, e.delay_oversampling)
        out["delay"].append(tau)
        out["aod"].append(float(beamspace.sine_to_angle(s_t, e.aod_branch)))
        out["aoa"].append(float(beamspace.sine_to_angle(s_r, e.aoa_branch)))
        out["gain"].append(delay_estimation.gain_from_delay(coef, tau, 1.0, Ts))
    return ChannelEstimate(np.array(out["delay"]), np.array(out["aod"]), np.array(out["aoa"]),
                           np.array(out["gain"], dtype=complex))


class LAMPEstimator(Estimator):
    """Unrolled sparse coarse stage -> SAGE -> LOS / EXIP-LMA (condition 'unknown' by default)."""

    name = "lamp"
    requires_oracle = False

    def __init__(self, checkpoint: str, condition: str = "unknown", threads: int = 1, continue_search: bool = False):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.net = GroupALISTA(**ck["model_kwargs"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.condition = condition
        self.continue_search = continue_search  # False: only LAMP proposals; True: then the correlation search (42)
        torch.set_num_threads(threads)

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        from ..localization.pipeline import localize
        cfg = context.config
        t0 = time.perf_counter()
        yn, Z, Ur = operator_inputs(y, context)
        with torch.no_grad():
            x = self.net(torch.from_numpy(yn)[None], torch.from_numpy(Z)[None], torch.from_numpy(Ur))[0].numpy()
        # learned front end proposes atoms; acceptance uses the frozen baseline's calibrated test (41) on the residual
        props = peak_proposals(x, cfg)
        # no proposal (the unrolled estimate shrank everything to zero): fall back to the correlation search (42)
        coarse = coarse_estimate(y, context, proposals=props or None, continue_search=self.continue_search)
        t1 = time.perf_counter()
        if not coarse.num_paths:
            return PositionEstimate.failed("no path")
        sr = sage(y, context, coarse, max_iter=cfg.estimator.sage_max_iter, tol=cfg.estimator.sage_tol)
        t2 = time.perf_counter()
        cond = cfg.scenario.condition if self.condition == "auto" else self.condition
        out = localize(sr.estimate, context, cond)
        out.info.update({"coarse": coarse, "sage": sr,
                         "timing": {"coarse": t1 - t0, "sage": t2 - t1, "localize": time.perf_counter() - t2}})
        return out

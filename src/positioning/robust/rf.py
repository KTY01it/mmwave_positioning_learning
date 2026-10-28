"""RF / model-mismatch layer for simulation stress tests (docs/robustness_completion_plan.md, Sec. 5).

The noiseless observation is rebuilt from the true channel with an impaired hardware model; the noise realisation and
N0 of the clean sample are kept. The estimators keep the ideal model. Magnitudes are design-chosen stress levels:
these are NOT hardware experiments. Separate from the simulator and from positioning.impairments.

    P  per-chain phase offsets     level = sigma [deg]          s -> s * e^{j xi},  y -> y * e^{j psi}
    A  gain imbalance              level = sigma [dB]           per-chain amplitudes 10^{sigma u / 20}, signal path only
    F  carrier frequency offset    level = df / fc [ppm]        e^{j 2 pi df t} over G sequential OFDM symbols of
                                                                N + N_CP samples, exact inter-carrier leakage
    T  timing offset               level = delta [samples]      e^{-j 2 pi n delta / N}
    S  array response perturbation level = sigma / lambda_c     Gaussian element-position errors in the array plane,
                                                                both arrays, per path and subcarrier

One device (unit draw) is scaled by the level, so the levels of one device are nested.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from ..system import channel as CH
from .data import NAMESPACE

N_CP = 5
N_DEVICES = 20
LEVELS = {"P": (2.0, 5.0, 10.0, 20.0), "A": (0.25, 0.5, 1.0, 2.0), "F": (0.1, 0.3, 1.0, 3.0),
          "T": (0.01, 0.03, 0.1, 0.3), "S": (0.01, 0.02, 0.05, 0.1)}
UNITS = {"P": "deg", "A": "dB", "F": "ppm", "T": "sample", "S": "lambda"}


def device(index: int, nt: int, nr: int) -> dict:
    """Unit-variance draws of device `index mod N_DEVICES` (independent of the sample)."""
    d = int(index) % N_DEVICES
    ent = int.from_bytes(hashlib.sha256(f"{NAMESPACE}/r4_device".encode()).digest(), "big")
    rng = np.random.default_rng(np.random.SeedSequence(entropy=ent, spawn_key=(d,)))
    return {"id": d, "p_tx": rng.standard_normal(nt), "p_rx": rng.standard_normal(nr),
            "a_tx": rng.standard_normal(nt), "a_rx": rng.standard_normal(nr),
            "s_tx": rng.standard_normal((2, nt)), "s_rx": rng.standard_normal((2, nr))}


def _path_channels(sample):
    """[K, N, Nr, Nt]: one channel per true path, built by the simulator's own channel function."""
    p, cfg = sample.gt.paths, sample.context.config
    return np.stack([CH.build_channel_from_config(
        CH.PathParameters(**{f.name: getattr(p, f.name)[k:k + 1] for f in dataclasses.fields(p)}), cfg)
        for k in range(p.num_paths)])


def impaired_noiseless(sample, kind: str, level: float, dev: dict) -> np.ndarray:
    """Noiseless observation [N, G, Nr] of `sample` under impairment `kind` at `level`."""
    ctx = sample.context
    cfg = ctx.config
    s = ctx.s                                                     # [N, G, Nt]
    N, G, _ = s.shape
    if kind == "P":
        sig = np.deg2rad(level)
        return np.einsum("nrt,ngt->ngr", sample.h, s * np.exp(1j * sig * dev["p_tx"])) * np.exp(1j * sig * dev["p_rx"])
    if kind == "A":
        return np.einsum("nrt,ngt->ngr", sample.h, s * 10 ** (level * dev["a_tx"] / 20)) * 10 ** (level * dev["a_rx"] / 20)
    clean = np.einsum("nrt,ngt->ngr", sample.h, s)
    if kind == "T":
        return clean * np.exp(-2j * np.pi * np.arange(N) * level / N)[:, None, None]
    if kind == "F":
        eps = level * 1e-6 * cfg.physics.fc * N / cfg.physics.bandwidth          # df in subcarrier spacings
        t = (np.arange(G)[None, :] * (N + N_CP) + N_CP + np.arange(N)[:, None])  # [m, g] sample index after the CP
        rot = np.exp(2j * np.pi * eps * t / N)
        return np.fft.fft(np.fft.ifft(clean, axis=0) * rot[:, :, None], axis=0)
    if kind == "S":
        p = sample.gt.paths
        k_n = 2 * np.pi / ctx.wavelengths                                        # [N]
        sig = level * cfg.physics.wavelength_c

        def pert(theta, u):                                                      # [K, N, Nant], sign as in Eq. (5)
            proj = np.sin(theta)[:, None] * u[0][None, :] + np.cos(theta)[:, None] * u[1][None, :]
            return np.exp(-1j * sig * k_n[None, :, None] * proj[:, None, :])
        H = np.einsum("knrt,knr,knt->nrt", _path_channels(sample), pert(p.aoa, dev["s_rx"]), pert(p.aod, dev["s_tx"]).conj())
        return np.einsum("nrt,ngt->ngr", H, s)
    raise ValueError(f"unknown impairment {kind!r}")


def impair(sample, kind: str, level: float, dev: dict | None = None, index: int = 0):
    """`sample` (drawn with keep_noiseless=True) with the impaired signal and the SAME noise realisation and N0."""
    if sample.noiseless is None:
        raise ValueError("the sample must be drawn with keep_noiseless=True")
    cfg = sample.context.config
    dev = dev or device(index, cfg.array.num_tx, cfg.array.num_rx)
    noise = sample.y - sample.noiseless
    return dataclasses.replace(sample, y=impaired_noiseless(sample, kind, level, dev) + noise)

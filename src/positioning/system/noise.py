"""AWGN and the SNR definition of Shahmansoori et al., Eq. (68).

SNR = E||diag{Omega[n]} vec{h_check[n]}||^2 / E||vec{n_check[n]}||^2.

With unitary U_Tx, U_Rx, Omega[n] h_check[n] stacks H[n] F^{(g)}[n] x^{(g)}[n] over g, and the noise
vector has N*G*Nr entries of variance N0, hence (F7)

    N0 = sum_{n,g} ||H[n] F^{(g)}[n] x^{(g)}[n]||^2 / (N * G * Nr * SNR).

Mode "target_snr" uses the energy of the realised noiseless signal; mode "n0" uses a fixed N0
(constant transmit power, e.g. Fig. 2 and main.m's sigma = 0.1).
"""
from __future__ import annotations

import numpy as np


def complex_awgn(shape, n0: float, rng: np.random.Generator) -> np.ndarray:
    """Circular complex Gaussian, variance n0 (n0/2 per real dimension)."""
    return np.sqrt(n0 / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def noise_variance_from_snr(signal: np.ndarray, snr_db: float) -> float:
    """N0 such that Eq. (68) equals snr_db for this noiseless signal (any shape, all entries count)."""
    energy = float(np.sum(np.abs(signal) ** 2))
    return energy / (signal.size * 10 ** (snr_db / 10))


def snr_db_from_n0(signal: np.ndarray, n0: float) -> float:
    return float(10 * np.log10(np.sum(np.abs(signal) ** 2) / (signal.size * n0)))


def measured_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Empirical ratio ||signal||^2 / ||noise||^2 in dB."""
    return float(10 * np.log10(np.sum(np.abs(signal) ** 2) / np.sum(np.abs(noise) ** 2)))


def add_awgn_from_target_snr(signal: np.ndarray, snr_db: float, config, rng: np.random.Generator,
                             noise_unit=None):
    """Return (y, noise, n0) according to config.noise.

    ``noise_unit`` (optional, same shape as signal) is a unit-variance complex Gaussian draw that is
    scaled by sqrt(n0); this lets parity tests inject the exact random numbers used by MATLAB.
    """
    nc = config.noise
    if nc.mode == "target_snr":
        n0 = noise_variance_from_snr(signal, snr_db)
    elif nc.mode == "n0":
        if nc.n0 is None:
            raise ValueError("noise.mode='n0' requires noise.n0")
        n0 = float(nc.n0)
    else:
        raise ValueError(f"unknown noise mode {nc.mode!r}")
    if noise_unit is None:
        noise = complex_awgn(signal.shape, n0, rng)
    else:
        noise = np.sqrt(n0) * np.asarray(noise_unit)
    return signal + noise, noise, n0

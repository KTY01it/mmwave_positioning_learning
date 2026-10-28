"""Frequency-dependent ULA response vectors (Shahmansoori et al., Eq. (5)).

a_n(theta)[i] = exp(-j (2 pi / lambda_n) d sin(theta) i) / sqrt(N_ant)

with element index i in {-(N-1)/2, ..., (N-1)/2} ("centered", Eq. 5) or {0, ..., N-1}
("zero", getResponse.m). lambda_n = c / (n/(N Ts) + fc), n = 0..N-1 (F15).
"""
from __future__ import annotations

import numpy as np


def subcarrier_frequencies(fc: float, bandwidth: float, num_subcarriers: int) -> np.ndarray:
    """f_n = fc + n/(N Ts) = fc + n B / N, n = 0..N-1."""
    n = np.arange(num_subcarriers)
    return fc + n * bandwidth / num_subcarriers


def wavelength(fc: float, subcarrier_freq=None, c: float = 299_792_000.0):
    """lambda_n = c / f_n. With subcarrier_freq=None returns lambda_c."""
    if subcarrier_freq is None:
        return c / fc
    return c / np.asarray(subcarrier_freq, dtype=float)


def subcarrier_wavelengths(c, fc, bandwidth, num_subcarriers, frequency_dependent=True) -> np.ndarray:
    """[N] wavelengths; constant lambda_c when the array is modelled as narrowband."""
    if not frequency_dependent:
        return np.full(num_subcarriers, c / fc)
    return wavelength(fc, subcarrier_frequencies(fc, bandwidth, num_subcarriers), c)


def element_indices(num_antennas: int, index_origin: str = "centered") -> np.ndarray:
    i = np.arange(num_antennas, dtype=float)
    if index_origin == "centered":
        return i - (num_antennas - 1) / 2
    if index_origin == "zero":
        return i
    raise ValueError(f"index_origin must be 'centered' or 'zero', got {index_origin!r}")


def _phase_slope(theta, wavelength, spacing):
    """(2 pi / lambda) d sin(theta), broadcast over theta and wavelength."""
    theta = np.asarray(theta, dtype=float)
    lam = np.asarray(wavelength, dtype=float)
    return 2 * np.pi * spacing * np.sin(theta)[..., None] / lam[None, ...] if theta.ndim and lam.ndim \
        else 2 * np.pi * spacing * np.sin(theta) / lam


def ula_response(theta, num_antennas: int, wavelength, spacing: float, index_origin: str = "centered"):
    """Unit-norm ULA response.

    Shapes: theta [...T], wavelength [...L] -> [...T, ...L, num_antennas] (scalars drop out).
    Example: theta [K], wavelength [N] -> [K, N, Nant].
    """
    k = _phase_slope(theta, wavelength, spacing)
    i = element_indices(num_antennas, index_origin)
    return np.exp(-1j * np.asarray(k)[..., None] * i) / np.sqrt(num_antennas)


def ula_response_derivative(theta, num_antennas: int, wavelength, spacing: float,
                            index_origin: str = "centered"):
    """d a / d theta = -j (2 pi / lambda) d cos(theta) i * a  (cf. D_Tx in Eq. 81)."""
    theta_arr = np.asarray(theta, dtype=float)
    a = ula_response(theta, num_antennas, wavelength, spacing, index_origin)
    lam = np.asarray(wavelength, dtype=float)
    if theta_arr.ndim and lam.ndim:
        g = 2 * np.pi * spacing * np.cos(theta_arr)[..., None] / lam[None, ...]
    else:
        g = 2 * np.pi * spacing * np.cos(theta_arr) / lam
    i = element_indices(num_antennas, index_origin)
    return -1j * np.asarray(g)[..., None] * i * a

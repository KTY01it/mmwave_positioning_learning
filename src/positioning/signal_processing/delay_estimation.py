"""Coarse delay and gain from the DCS-SOMP coefficients, Eqs. (49)-(54); main.m L94-99.

Model (51): h_check^{(k)} = h~_k z^{(k)} a(tau_k) + v,  a(tau)[n] = exp(-j 2 pi n tau / (N Ts)).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def delay_steering(tau, num_subcarriers: int, sampling_period: float) -> np.ndarray:
    n = np.arange(num_subcarriers)
    return np.exp(-2j * np.pi * n * tau / (num_subcarriers * sampling_period))


def phase_slope_delay(coef: np.ndarray, sampling_period: float) -> float:
    """main.m: -mean(diff(phase(h)))*N*Ts/(2 pi), wrapped into [0, N Ts). MATLAB phase() = unwrap(angle())."""
    N = coef.size
    tau = -np.mean(np.diff(np.unwrap(np.angle(coef)))) * N * sampling_period / (2 * np.pi)
    if tau < 0:
        tau += N * sampling_period
    return float(tau)


def periodogram_delay(coef: np.ndarray, sampling_period: float, oversampling: int = 16,
                      refine: bool = True) -> float:
    """Eq. (54): argmax_tau |a^H(tau) h|^2 over [0, N Ts): zero-padded FFT then bounded Brent."""
    N = coef.size
    L = N * oversampling
    # a^H(tau_m) h = sum_n h[n] exp(+j 2 pi n m / L) with tau_m = m Ts / oversampling
    spec = np.abs(np.fft.ifft(coef, L) * L) ** 2
    m = int(np.argmax(spec))
    tau0 = m * sampling_period / oversampling
    if not refine:
        return float(tau0)
    f = lambda t: -np.abs(np.vdot(delay_steering(t, N, sampling_period), coef)) ** 2
    step = sampling_period / oversampling
    res = minimize_scalar(f, bounds=(tau0 - step, tau0 + step), method="bounded",
                          options={"xatol": sampling_period * 1e-7})
    return float(np.mod(res.x, N * sampling_period))


def gain_from_delay(coef: np.ndarray, tau: float, z: complex, sampling_period: float) -> complex:
    """Eq. (53): h~ = a^H(tau) h / (z N)."""
    N = coef.size
    return complex(np.vdot(delay_steering(tau, N, sampling_period), coef) / (z * N))

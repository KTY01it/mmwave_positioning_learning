"""Wideband mmWave MIMO channel, Shahmansoori et al. Eqs. (1)-(4), (66)-(67).

H[n] = A_Rx[n] Gamma[n] A_Tx^H[n],
Gamma[n] = sqrt(Nt Nr) diag{ h_k / sqrt(rho_k) * exp(-j 2 pi n tau_k / (N Ts)) }.

We store ``gain = h_k / sqrt(rho_k)`` and ``gain_tilde = sqrt(Nt Nr) * gain`` (= h~_k, the channel
coefficient that is estimated, Sec. II-B).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arrays import ula_response, subcarrier_wavelengths
from .geometry import PathGeometry


@dataclass
class PathParameters:
    """Ground-truth channel parameters of K paths (index 0 = LOS when present)."""

    delay: np.ndarray  # [K] s
    aod: np.ndarray  # [K] rad
    aoa: np.ndarray  # [K] rad
    gain: np.ndarray  # [K] complex, h_k / sqrt(rho_k)
    gain_tilde: np.ndarray  # [K] complex, sqrt(Nt Nr) h_k / sqrt(rho_k)
    rho: np.ndarray  # [K] path loss (linear, >= 1 for physical losses)
    is_los: np.ndarray  # [K] bool

    @property
    def num_paths(self) -> int:
        return int(self.delay.size)

    def as_eta(self) -> np.ndarray:
        """Channel parameter vector eta, Eqs. (7)-(8): per path [tau, theta_Tx, theta_Rx, Re h~, Im h~]."""
        return np.stack([self.delay, self.aod, self.aoa, self.gain_tilde.real, self.gain_tilde.imag],
                        axis=1).ravel()


# --------------------------------------------------------------------------------------
# path loss and gains
# --------------------------------------------------------------------------------------
def atmospheric_attenuation(distance, db_per_km: float):
    """xi^2(d) as a linear power factor (<= 1)."""
    return 10.0 ** (-db_per_km * np.asarray(distance, dtype=float) / 1000.0 / 10.0)


def path_loss(distance, d2, is_los, wavelength_c, *, sigma0_sq=None, poisson_density=1 / 7,
              atmospheric_db_per_km=16.0):
    """rho_k from Eqs. (66)-(67). Returns rho (>0); the power gain of the path is 1/rho.

    1/rho_0 = xi^2(d_0) (lambda_c / (4 pi d_0))^2
    1/rho_k = sigma_0^2 P0(d_k2) xi^2(d_k) (lambda_c / (4 pi d_k))^2,  P0(d) = (gamma_r d)^2 exp(-gamma_r d)
    """
    distance = np.asarray(distance, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    is_los = np.asarray(is_los, dtype=bool)
    fspl = (wavelength_c / (4 * np.pi * distance)) ** 2
    inv_rho = atmospheric_attenuation(distance, atmospheric_db_per_km) * fspl
    p0 = (poisson_density * d2) ** 2 * np.exp(-poisson_density * d2)
    sig = np.ones_like(distance) if sigma0_sq is None else np.asarray(sigma0_sq, dtype=float)
    inv_rho = np.where(is_los, inv_rho, inv_rho * sig * p0)
    return 1.0 / inv_rho


def draw_path_gains(geometry: PathGeometry, cfg, rng: np.random.Generator | None,
                    phases=None, sigma0_db=None):
    """Return (gain = h_k/sqrt(rho_k), rho, sigma0_db) according to cfg.channel.

    ``phases``/``sigma0_db`` may be injected (parity tests); otherwise drawn from ``rng``.
    Draw order (fixed for reproducibility): phases [K], then sigma0_db [K].
    """
    ch = cfg.channel
    K = geometry.num_paths
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, K) if ch.gain_phase == "random" else np.zeros(K)
    phases = np.asarray(phases, dtype=float)
    if ch.gain_model == "constant":
        rho = np.ones(K)
        gain = ch.constant_gain * np.exp(1j * phases)
        return gain, rho, np.full(K, np.nan)
    if ch.gain_model != "pathloss":
        raise ValueError(f"unknown gain_model {ch.gain_model!r}")
    if sigma0_db is None:
        if ch.reflection_loss_random:
            sigma0_db = ch.reflection_loss_db_mean + ch.reflection_loss_db_std * rng.standard_normal(K)
        else:
            sigma0_db = np.full(K, ch.reflection_loss_db_mean)
    sigma0_db = np.where(geometry.is_los, np.nan, np.asarray(sigma0_db, dtype=float))
    rho = path_loss(
        geometry.distance, geometry.d2, geometry.is_los, cfg.physics.wavelength_c,
        sigma0_sq=np.where(geometry.is_los, 1.0, 10 ** (np.nan_to_num(sigma0_db) / 10)),
        poisson_density=ch.poisson_density,
        atmospheric_db_per_km=ch.atmospheric_attenuation_db_per_km,
    )
    gain = np.exp(1j * phases) / np.sqrt(rho)
    return gain, rho, sigma0_db


def make_path_parameters(geometry: PathGeometry, gain, rho, num_tx: int, num_rx: int) -> PathParameters:
    gain = np.asarray(gain, dtype=complex)
    return PathParameters(
        delay=geometry.delay.copy(), aod=geometry.aod.copy(), aoa=geometry.aoa.copy(),
        gain=gain, gain_tilde=np.sqrt(num_tx * num_rx) * gain, rho=np.asarray(rho, dtype=float),
        is_los=geometry.is_los.copy(),
    )


# --------------------------------------------------------------------------------------
# channel matrix
# --------------------------------------------------------------------------------------
def delay_phasors(delay, num_subcarriers: int, sampling_period: float) -> np.ndarray:
    """exp(-j 2 pi n tau_k / (N Ts)), shape [K, N]."""
    n = np.arange(num_subcarriers)
    return np.exp(-2j * np.pi * np.outer(np.atleast_1d(delay), n) / (num_subcarriers * sampling_period))


def build_channel(paths: PathParameters, *, fc: float, bandwidth: float, num_subcarriers: int,
                  num_tx: int, num_rx: int, c: float, spacing: float,
                  index_origin: str = "centered", frequency_dependent: bool = True) -> np.ndarray:
    """H with shape [N, Nr, Nt] (Eq. 1). ``spacing`` is the absolute element spacing d [m]."""
    lam = subcarrier_wavelengths(c, fc, bandwidth, num_subcarriers, frequency_dependent)  # [N]
    a_tx = ula_response(paths.aod, num_tx, lam, spacing, index_origin)  # [K, N, Nt]
    a_rx = ula_response(paths.aoa, num_rx, lam, spacing, index_origin)  # [K, N, Nr]
    gamma = paths.gain_tilde[:, None] * delay_phasors(paths.delay, num_subcarriers, 1.0 / bandwidth)  # [K, N]
    return np.einsum("kn,knr,knt->nrt", gamma, a_rx, a_tx.conj())


def build_channel_from_config(paths: PathParameters, cfg) -> np.ndarray:
    ph, ar = cfg.physics, cfg.array
    return build_channel(paths, fc=ph.fc, bandwidth=ph.bandwidth, num_subcarriers=ph.num_subcarriers,
                         num_tx=ar.num_tx, num_rx=ar.num_rx, c=ph.speed_of_light,
                         spacing=cfg.antenna_spacing, index_origin=ar.index_origin,
                         frequency_dependent=ph.frequency_dependent_array)

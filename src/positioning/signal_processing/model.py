"""Receiver-side forward model of the noiseless observation mu(eta), Eq. (6) / (11).

mu^{(g)}[n] = sum_k h~_k exp(-j 2 pi n tau_k/(N Ts)) a_Rx,n(theta_Rx,k) a_Tx,n(theta_Tx,k)^H s^{(g)}[n]

It reuses the system's array response (system/arrays.py) and only depends on what the receiver knows
(ObservationContext): pilots s = F x, wavelengths, array geometry. Used by SAGE (Eqs. 55-60), EXIP
weighting and the FIM (Eq. 69 summed over g, F14).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..system.arrays import ula_response, ula_response_derivative


@dataclass
class ChannelEstimate:
    """Per-path channel parameters (estimated or true). Units: s, rad, complex h~."""

    delay: np.ndarray
    aod: np.ndarray
    aoa: np.ndarray
    gain_tilde: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def num_paths(self) -> int:
        return int(np.size(self.delay))

    def as_eta(self) -> np.ndarray:
        return np.stack([self.delay, self.aod, self.aoa, self.gain_tilde.real, self.gain_tilde.imag],
                        axis=1).ravel()

    @classmethod
    def from_eta(cls, eta, info=None) -> "ChannelEstimate":
        e = np.asarray(eta, dtype=float).reshape(-1, 5)
        return cls(e[:, 0].copy(), e[:, 1].copy(), e[:, 2].copy(), e[:, 3] + 1j * e[:, 4], info or {})

    @classmethod
    def from_paths(cls, paths) -> "ChannelEstimate":
        return cls(paths.delay.copy(), paths.aod.copy(), paths.aoa.copy(), paths.gain_tilde.copy())

    def sorted_by_delay(self) -> "ChannelEstimate":
        o = np.argsort(self.delay)
        return ChannelEstimate(self.delay[o], self.aod[o], self.aoa[o], self.gain_tilde[o], dict(self.info))

    def subset(self, idx) -> "ChannelEstimate":
        idx = np.atleast_1d(idx)
        return ChannelEstimate(self.delay[idx], self.aod[idx], self.aoa[idx], self.gain_tilde[idx], dict(self.info))


class ObservationModel:
    def __init__(self, context):
        cfg = context.config
        self.s = context.s  # [N, G, Nt]
        self.lam = np.asarray(context.wavelengths)  # [N]
        self.N, self.G, self.nt = self.s.shape
        self.nr = cfg.array.num_rx
        self.d = cfg.antenna_spacing
        self.origin = cfg.array.index_origin
        self.Ts = cfg.physics.sampling_period
        self.n = np.arange(self.N)

    # ---- building blocks; leading candidate axis C when the argument is an array --------
    def tx_projection(self, aod):
        """a_Tx,n(theta)^H s^{(g)}[n]: scalar -> [N, G]; [C] -> [C, N, G]."""
        a = ula_response(aod, self.nt, self.lam, self.d, self.origin)
        return np.einsum("...nt,ngt->...ng", a.conj(), self.s)

    def tx_projection_derivative(self, aod):
        da = ula_response_derivative(aod, self.nt, self.lam, self.d, self.origin)
        return np.einsum("...nt,ngt->...ng", da.conj(), self.s)

    def rx_response(self, aoa):
        """[N, Nr] (or [C, N, Nr])."""
        return ula_response(aoa, self.nr, self.lam, self.d, self.origin)

    def rx_response_derivative(self, aoa):
        return ula_response_derivative(aoa, self.nr, self.lam, self.d, self.origin)

    def delay_phasor(self, tau):
        """[N] (or [C, N])."""
        tau = np.asarray(tau, dtype=float)
        return np.exp(-2j * np.pi * np.multiply.outer(tau, self.n) / (self.N * self.Ts))

    # ---- signals ----------------------------------------------------------------------
    def path_basis(self, tau, aod, aoa):
        """Gain-free single-path signal b[n,g,r], shape [N, G, Nr]."""
        e = self.delay_phasor(tau)
        return e[:, None, None] * self.tx_projection(aod)[:, :, None] * self.rx_response(aoa)[:, None, :]

    def path_signal(self, tau, aod, aoa, gain):
        return gain * self.path_basis(tau, aod, aoa)

    def mean(self, est: ChannelEstimate) -> np.ndarray:
        out = np.zeros((self.N, self.G, self.nr), dtype=complex)
        for k in range(est.num_paths):
            out += self.path_signal(est.delay[k], est.aod[k], est.aoa[k], est.gain_tilde[k])
        return out

    def jacobian(self, est: ChannelEstimate) -> np.ndarray:
        """D = d mu / d eta^T, shape [N*G*Nr, 5K], eta ordering of Eq. (8)."""
        cols = []
        for k in range(est.num_paths):
            tau, ad, aa, h = est.delay[k], est.aod[k], est.aoa[k], est.gain_tilde[k]
            e = self.delay_phasor(tau)[:, None, None]
            tp = self.tx_projection(ad)[:, :, None]
            dtp = self.tx_projection_derivative(ad)[:, :, None]
            ar = self.rx_response(aa)[:, None, :]
            dar = self.rx_response_derivative(aa)[:, None, :]
            de = (-2j * np.pi * self.n / (self.N * self.Ts))[:, None, None]
            base = e * tp * ar
            cols += [h * de * base, h * e * dtp * ar, h * e * tp * dar, base, 1j * base]
        return np.stack([c.ravel() for c in cols], axis=1)

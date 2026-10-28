"""Beamspace representation, Shahmansoori et al. Eqs. (31)-(40).

Dictionary columns are ULA responses at lambda_c on a grid of spatial frequencies
psi_m = (d / lambda_c) sin(theta_m), i.e. column m = exp(-j 2 pi psi_m i) / sqrt(Nant) (F5).

grid "centered" (paper): psi_m = (m - (Nb+1)/2) / Nb,  m = 1..Nb    (unitary for any parity when Nb = Nant, F4)
grid "github"  (main.m): psi_m = (m - Nb/2 - 1) / Nb,  m = 1..Nb    (aa = 2*psi with d = lambda/2)

Observation stacking (Eq. 36): y_check[n] = vec over (g outer, r inner) of y^{(g)}[n].
Sensing matrix (Eqs. 37-39): Omega[n] = kron(Z[n]^T, U_Rx), Z[n] = U_Tx^H [F x]^{(1..G)}[n];
column j = it * Nbr + ir, i.e. vec(H_check) is column-major (Tx index outer).
"""
from __future__ import annotations

import numpy as np

from ..system.arrays import element_indices
from ..system.geometry import wrap_angle


# --------------------------------------------------------------------------------------
# dictionary
# --------------------------------------------------------------------------------------
def dictionary_grid(num_antennas: int, oversampling: int = 1, grid: str = "centered") -> np.ndarray:
    nb = int(oversampling * num_antennas)
    m = np.arange(1, nb + 1)
    if grid == "centered":
        return (m - (nb + 1) / 2) / nb
    if grid == "github":
        return (m - nb / 2 - 1) / nb
    raise ValueError(f"unknown dictionary grid {grid!r}")


def dictionary(num_antennas: int, wavelength_c: float, spacing: float, index_origin: str = "centered",
               oversampling: int = 1, grid: str = "centered", normalize: bool = True) -> np.ndarray:
    """U [Nant, Nb]. normalize=False reproduces main.m (columns of norm sqrt(Nant))."""
    psi = dictionary_grid(num_antennas, oversampling, grid)
    i = element_indices(num_antennas, index_origin)
    U = np.exp(-2j * np.pi * np.outer(i, psi)) / np.sqrt(num_antennas)
    return U if normalize else U * np.sqrt(num_antennas)


def grid_sine(num_antennas, wavelength_c, spacing, oversampling=1, grid="centered") -> np.ndarray:
    """sin(theta_m) of every dictionary column."""
    return dictionary_grid(num_antennas, oversampling, grid) * wavelength_c / spacing


def sine_to_angle(sine, branch: str = "front"):
    """Resolve the ULA front/back ambiguity (F9). front: arcsin in [-pi/2, pi/2]; back: pi - arcsin."""
    a = np.arcsin(np.clip(sine, -1.0, 1.0))
    if branch == "front":
        return a
    if branch == "back":
        return wrap_angle(np.pi - a)
    raise ValueError(f"unknown branch {branch!r}")


def dictionaries_from_config(cfg):
    """(U_Tx, U_Rx, sinTx grid, sinRx grid) according to cfg.estimator."""
    e, ar = cfg.estimator, cfg.array
    lam, d = cfg.physics.wavelength_c, cfg.antenna_spacing
    kw = dict(index_origin=ar.index_origin, oversampling=e.dictionary_oversampling, grid=e.dictionary_grid)
    Ut = dictionary(ar.num_tx, lam, d, normalize=e.dictionary_normalize, **kw)
    Ur = dictionary(ar.num_rx, lam, d, normalize=e.dictionary_normalize, **kw)
    st = grid_sine(ar.num_tx, lam, d, e.dictionary_oversampling, e.dictionary_grid)
    sr = grid_sine(ar.num_rx, lam, d, e.dictionary_oversampling, e.dictionary_grid)
    return Ut, Ur, st, sr


# --------------------------------------------------------------------------------------
# beamspace channel and observation
# --------------------------------------------------------------------------------------
def beamspace_channel(H: np.ndarray, Ut: np.ndarray, Ur: np.ndarray) -> np.ndarray:
    """H_check[n] = U_Rx^H H[n] U_Tx (Eq. 31), shape [N, Nbr, Nbt]."""
    return np.einsum("rb,nrt,tc->nbc", Ur.conj(), H, Ut)


def vec_beamspace(Hc: np.ndarray) -> np.ndarray:
    """h_check[n] = vec(H_check[n]) column-major (Eq. 40), shape [N, Nbt*Nbr]."""
    return np.transpose(Hc, (0, 2, 1)).reshape(Hc.shape[0], -1)


def stack_observation(y: np.ndarray) -> np.ndarray:
    """[N, G, Nr] -> [N, G*Nr] (Eq. 36; equals reshape(y(:,:,n), Nr*Ns, 1) in MATLAB)."""
    return y.reshape(y.shape[0], -1)


def beam_projections(s: np.ndarray, Ut: np.ndarray) -> np.ndarray:
    """Z[n]^T with Z[n] = U_Tx^H [s^{(1)}[n] ... s^{(G)}[n]] (Eq. 39). Shape [N, G, Nbt]."""
    return np.einsum("ngt,tb->ngb", s, Ut.conj())


def sensing_matrix(s: np.ndarray, Ut: np.ndarray, Ur: np.ndarray) -> np.ndarray:
    """Explicit Omega [N, G*Nr, Nbt*Nbr] (Eqs. 37-38). Memory heavy; use SensingOperator instead."""
    Zt = beam_projections(s, Ut)
    return np.stack([np.kron(Zt[n], Ur) for n in range(s.shape[0])])


def split_index(j, nbr: int):
    """Flat column index -> (Tx dictionary index, Rx dictionary index), 0-based (Eq. 43)."""
    j = np.asarray(j)
    return j // nbr, j % nbr


class SensingOperator:
    """Matrix-free Omega[n] exploiting Omega[n] = kron(Z[n]^T, U_Rx)."""

    def __init__(self, s: np.ndarray, Ut: np.ndarray, Ur: np.ndarray):
        self.Zt = beam_projections(s, Ut)  # [N, G, Nbt]
        self.Ur = Ur
        self.N, self.G, self.nbt = self.Zt.shape
        self.nr, self.nbr = Ur.shape
        self.num_columns = self.nbt * self.nbr
        self._norms = (np.linalg.norm(self.Zt, axis=1)[:, :, None] *
                       np.linalg.norm(Ur, axis=0)[None, None, :]).reshape(self.N, -1)

    def correlate(self, r: np.ndarray) -> np.ndarray:
        """Omega[n]^H r[n] for all n: r [N, G*Nr] -> [N, Nbt*Nbr]."""
        R = r.reshape(self.N, self.G, self.nr)
        M = np.einsum("ngb,ngc->nbc", self.Zt.conj(), R @ self.Ur.conj())
        return M.reshape(self.N, -1)

    def column_norms(self) -> np.ndarray:
        return self._norms

    def columns(self, j: int) -> np.ndarray:
        """omega_j[n] for all n, shape [N, G*Nr]."""
        it, ir = divmod(int(j), self.nbr)
        return (self.Zt[:, :, it][:, :, None] * self.Ur[None, None, :, ir]).reshape(self.N, -1)

    def explicit(self) -> np.ndarray:
        return np.stack([np.kron(self.Zt[n], self.Ur) for n in range(self.N)])

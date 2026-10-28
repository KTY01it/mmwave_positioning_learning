"""Modified DCS-SOMP, Shahmansoori et al. Algorithm 1, Eqs. (41)-(48); DCSSOMP.m.

Two stopping rules:
* ``num_atoms``: fixed sparsity (DCSSOMP.m, GitHub reference);
* ``threshold``: stop when the residual energy captured by the newest atom,
  sum_n ||r_{t-1}[n] - r_t[n]||^2, falls below delta of Eq. (41). The failing atom is discarded (F10).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaincinv


@dataclass
class DCSSOMPResult:
    atoms: np.ndarray  # [K] flat dictionary column indices (0-based), detection order
    coefficients: np.ndarray  # [K, N] h_check^{(k)}[n] (Eq. 48)
    residual: np.ndarray  # [N, M]
    captured_energy: list = field(default_factory=list)  # per accepted atom
    threshold: float | None = None
    rejected_energy: float | None = None  # energy of the first rejected atom (threshold mode)

    @property
    def num_paths(self) -> int:
        return int(self.atoms.size)


def detection_threshold(n0: float, num_subcarriers: int, num_columns: int, pfa: float = 1e-3) -> float:
    """delta = N0 * P^{-1}(N, (1 - Pfa)^{1/(Nr Nt)})  (Eq. 41 with the regularised gamma, F10).

    The noise energy captured by one unit-norm column summed over N subcarriers is N0 * Gamma(N, 1);
    delta is the (1 - Pfa) quantile of its maximum over all ``num_columns`` columns.
    """
    return float(n0 * gammaincinv(num_subcarriers, (1.0 - pfa) ** (1.0 / num_columns)))


def dcs_somp(y: np.ndarray, operator, *, num_atoms: int | None = None, threshold: float | None = None,
             max_atoms: int | None = None, min_atoms: int = 0) -> DCSSOMPResult:
    """y [N, M] (stacked observations, Eq. 36); operator: beamspace.SensingOperator-like."""
    if num_atoms is None and threshold is None:
        raise ValueError("give num_atoms (fixed sparsity) or threshold (Eq. 41)")
    N, M = y.shape
    limit = num_atoms if num_atoms is not None else (max_atoms or M)
    norms = operator.column_norms()  # [N, J]
    R = y.astype(complex).copy()
    psis, omegas, betas, captured = [], [], [], []
    atoms: list[int] = []
    rejected = None
    for _ in range(limit):
        # Eq. (42): sum_n |omega_m^H r| / ||omega_m||
        cost = np.sum(np.abs(operator.correlate(R)) / norms, axis=0)
        j = int(np.argmax(cost))
        omega = operator.columns(j)  # [N, M]
        # Eq. (46): classical Gram-Schmidt against previous psi (with the original omega)
        psi = omega.copy()
        for p in psis:
            psi -= (np.sum(p.conj() * omega, axis=1) / np.sum(np.abs(p) ** 2, axis=1))[:, None] * p
        pn = np.sum(np.abs(psi) ** 2, axis=1)
        beta = np.sum(psi.conj() * R, axis=1) / pn  # Eq. (47)
        energy = float(np.sum(np.abs(beta) ** 2 * pn))
        if threshold is not None and num_atoms is None and energy < threshold and len(atoms) >= min_atoms:
            rejected = energy
            break
        R = R - beta[:, None] * psi
        atoms.append(j); psis.append(psi); omegas.append(omega); betas.append(beta); captured.append(energy)
    K = len(atoms)
    coef = np.zeros((K, N), dtype=complex)
    if K:
        proj = y - R  # = Psi beta, projection of y onto span of chosen columns
        cols = np.stack(omegas, axis=2)  # [N, M, K]
        for n in range(N):  # Eq. (48), equivalent to R^{-1} Q^H Psi beta (F11)
            coef[:, n] = np.linalg.lstsq(cols[n], proj[n], rcond=None)[0]
    return DCSSOMPResult(np.array(atoms, dtype=int), coef, R, captured, threshold, rejected)

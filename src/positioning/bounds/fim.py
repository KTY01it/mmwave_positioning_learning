"""FIM of the channel parameters, Eqs. (10)-(14), (69).

J_eta = (2/N0) Re{ D^H D },  D = d mu / d eta^T over all (n, g, r)   (F14: sums the G transmissions;
Appendix A writes the G = 1 case). Parameter ordering per path: [tau, theta_Tx, theta_Rx, Re h~, Im h~].
"""
from __future__ import annotations

import numpy as np

from ..signal_processing.model import ChannelEstimate, ObservationModel


def channel_fim(context, paths, n0: float | None = None) -> np.ndarray:
    """paths: ChannelEstimate or system PathParameters (true values). Returns J_eta [5K, 5K]."""
    est = paths if isinstance(paths, ChannelEstimate) else ChannelEstimate.from_paths(paths)
    D = ObservationModel(context).jacobian(est)
    n0 = context.n0 if n0 is None else n0
    return (2.0 / n0) * np.real(D.conj().T @ D)


def block_diagonal(J: np.ndarray, block: int = 5) -> np.ndarray:
    """Drop inter-path terms (the large-array/bandwidth approximation behind Eq. 29)."""
    out = np.zeros_like(J)
    for i in range(0, J.shape[0], block):
        out[i:i + block, i:i + block] = J[i:i + block, i:i + block]
    return out

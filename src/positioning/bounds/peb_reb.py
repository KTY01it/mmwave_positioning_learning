"""Position and rotation error bounds, Eqs. (27)-(28); convenience wrapper for a simulated sample."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crb import channel_crb, position_fim
from .fim import block_diagonal, channel_fim


def peb(J_pos: np.ndarray) -> float:
    """sqrt(tr [J^{-1}]_{1:2,1:2}) (Eq. 27)."""
    C = np.linalg.inv(J_pos)
    return float(np.sqrt(C[0, 0] + C[1, 1]))


def reb(J_pos: np.ndarray) -> float:
    """sqrt([J^{-1}]_{3,3}) (Eq. 28)."""
    return float(np.sqrt(np.linalg.inv(J_pos)[2, 2]))


@dataclass
class Bounds:
    peb: float
    reb: float
    crb_delay: np.ndarray
    crb_aod: np.ndarray
    crb_aoa: np.ndarray
    J_eta: np.ndarray
    J_pos: np.ndarray


def compute_bounds(sample, approximate: bool = False) -> Bounds:
    """Bounds at the true parameters of a SimulationOutput (uses its pilots and N0)."""
    J = channel_fim(sample.context, sample.gt.paths)
    if approximate:
        J = block_diagonal(J)
    c = sample.context.config.physics.speed_of_light
    Jp = position_fim(J, sample.gt, c)
    cr = channel_crb(J)
    return Bounds(peb(Jp), reb(Jp), cr["delay"], cr["aod"], cr["aoa"], J, Jp)

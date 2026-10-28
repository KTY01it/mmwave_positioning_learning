"""CRB of channel parameters and FIM in the position domain, Eqs. (9), (15)-(16)."""
from __future__ import annotations

import numpy as np

from ..system import geometry


def channel_crb(J_eta: np.ndarray) -> dict:
    """sqrt(CRB) of tau_k, theta_Tx,k, theta_Rx,k from J_eta^{-1} (arrays over paths)."""
    C = np.linalg.inv(J_eta)
    d = np.sqrt(np.clip(np.diag(C), 0, None)).reshape(-1, 5)
    return {"delay": d[:, 0], "aod": d[:, 1], "aoa": d[:, 2], "gain_re": d[:, 3], "gain_im": d[:, 4]}


def position_fim(J_eta: np.ndarray, gt_or_geometry, c: float) -> np.ndarray:
    """J_eta_bar = T J_eta T^T (Eq. 15). ``gt_or_geometry``: simulator GroundTruth (or same attributes)."""
    g = gt_or_geometry
    has_los = bool(np.any(g.paths.is_los))
    T = geometry.transformation_matrix(g.bs_position, g.position, g.orientation, g.scatterers, has_los, c)
    return T @ J_eta @ T.T

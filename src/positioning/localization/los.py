"""LOS position and orientation, Eqs. (61)-(62)."""
from __future__ import annotations

import numpy as np

from ..signal_processing.model import ChannelEstimate
from ..system import geometry


def select_los_path(est: ChannelEstimate) -> int:
    """The path with the smallest delay is taken as LOS (main.m L102, Sec. IV-D)."""
    return int(np.argmin(est.delay))


def los_localize(est: ChannelEstimate, bs_position, c: float, path_index: int | None = None):
    """Return (p_hat, alpha_hat) using one path as LOS."""
    k = select_los_path(est) if path_index is None else path_index
    return geometry.los_position_from_parameters(est.delay[k], est.aod[k], est.aoa[k], bs_position, c)

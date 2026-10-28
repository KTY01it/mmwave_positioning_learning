"""T3: noise level without the simulator's N0.

Plug-in estimates from the observation (positioning.robust.noise, unchanged): E2 = tapered median, E3 = lower quartile.

Residual-floor ("resid") procedure: the noise level used by the stop and by the path tests is re-estimated from the
residual of the current model, N0_hat = median(cell powers of the residual) / ln 2, where the cells are the unitary
(delay, beam, receive-beam) transform of the pilot-derotated residual. Paths that are not yet in the model occupy
about G of the N*G*Nr cells and do not move the median. The global stop then compares the mean residual power with
its own median: T = ||r||^2 / N0_hat. Under the null hypothesis (residual = white noise) T / M = mean / (median / ln 2)
of M i.i.d. unit exponentials, whose distribution does not depend on N0. Its 0.99 quantile is 1 + A99 / sqrt(M) with
A99 = 2.45 (Monte Carlo, 20 000 draws each for M = 7 800 ... 41 600: 2.39 ... 2.48; the known-N0 chi-square test has
2.33). The path tests keep the frozen threshold Z with N0_hat in place of N0.
"""
from __future__ import annotations

import numpy as np

from ..robust.noise import cell_powers, n0_floor

A99 = 2.45
LN2 = float(np.log(2.0))


def plug_in(y, context, kind: str = "E2") -> float:
    return n0_floor(y, context, kind)


def residual_floor(r, context) -> float:
    return float(np.median(cell_powers(r, context)) / LN2)


def ratio_threshold(M: int) -> float:
    """Threshold of ||r||^2 / N0_hat at P_FA = 0.01 (unknown-scale stop)."""
    return M * (1.0 + A99 / np.sqrt(M))

"""Received signal, Eq. (6): y^{(g)}[n] = H[n] F^{(g)}[n] x^{(g)}[n] + n^{(g)}[n].

Shapes: H [N, Nr, Nt], F [N, G, Nt, Mt], X [N, G, Mt], noise / y [N, G, Nr].
"""
from __future__ import annotations

import numpy as np


def noiseless_observation(H: np.ndarray, F: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.einsum("nrt,ngtm,ngm->ngr", H, F, X)


def generate_observation(H: np.ndarray, F: np.ndarray, X: np.ndarray, noise=None) -> np.ndarray:
    y = noiseless_observation(H, F, X)
    return y if noise is None else y + noise

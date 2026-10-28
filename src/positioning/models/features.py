"""Input representations for learned estimators (master plan P4/P5). Receiver-side only: (y, context).

Angle-angle-delay correlation image (ADCPM-like, cf. [R5]): for every subcarrier the matched-filter output
Omega[n]^H y[n] on the unitary Nt x Nr beamspace grid (Eqs. 36-39), normalised per column so that the random beams
of each sample do not change the scale, divided by sqrt(N0) (noise-only entries are CN(0, 1)), then transformed to
the delay domain by an IFFT over subcarriers. Independent of the number of beams G.
"""
from __future__ import annotations

import numpy as np

from ..signal_processing import beamspace


def _dictionaries(cfg):
    lam, d, o = cfg.physics.wavelength_c, cfg.antenna_spacing, cfg.array.index_origin
    Ut = beamspace.dictionary(cfg.array.num_tx, lam, d, o, 1, "centered", True)
    Ur = beamspace.dictionary(cfg.array.num_rx, lam, d, o, 1, "centered", True)
    return Ut, Ur


def correlation_cube(y: np.ndarray, context) -> np.ndarray:
    """[N, Nt, Nr] complex: (Omega[n]^H y[n]) / (||omega_m[n]|| sqrt(N0))."""
    Ut, Ur = _dictionaries(context.config)
    op = beamspace.SensingOperator(context.s, Ut, Ur)
    c = op.correlate(beamspace.stack_observation(y)) / np.maximum(op.column_norms(), 1e-30)
    return (c / np.sqrt(context.n0)).reshape(y.shape[0], op.nbt, op.nbr)


def angle_delay_image(y: np.ndarray, context, num_delay_bins: int | None = None) -> np.ndarray:
    """float32 [D, Nt, Nr]: log(1 + |IFFT_n(correlation cube)| / sqrt(N)), D = N delay bins by default."""
    c = correlation_cube(y, context)
    N = c.shape[0]
    D = num_delay_bins or N
    t = np.fft.ifft(c, n=D, axis=0) * np.sqrt(D)  # unitary for D = N: noise stays CN(0, 1)
    return np.log1p(np.abs(t) / np.sqrt(N)).astype(np.float32)


# ---- target encoding (shared by every learned position estimator) -----------------------
P_CENTER = np.array([3.0, 0.15])
P_SCALE = np.array([1.0, 0.15])  # P3 v1 prior box: x in [2, 4], y in [0, 0.3]


def encode_pose(p, alpha) -> np.ndarray:
    p = (np.asarray(p, float) - P_CENTER) / P_SCALE
    return np.array([p[0], p[1], np.cos(alpha), np.sin(alpha)], dtype=np.float32)


def decode_pose(v):
    v = np.asarray(v, float)
    return v[:2] * P_SCALE + P_CENTER, float(np.arctan2(v[3], v[2]))

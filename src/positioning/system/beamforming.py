"""Transmit beams F^{(g)}[n] and symbols x^{(g)}[n] (Sec. II-A, Sec. V-A).

Shapes: F [N, G, Nt, Mt], x [N, G, Mt].

paper  : F^{(g)}[n] = F_RF^{(g)} F_BB, analog phases uniform on the unit circle and identical for all
         subcarriers, ||F||_F = 1; symbols x~^{(g)}[n] = exp(j phi_{g,n}).
GitHub : F(:,k,n) = exp(1j*rand(Nt,1)*2*pi) independently per (g, n), not normalised; x = 1.
"""
from __future__ import annotations

import numpy as np


def generate_beam_phases(cfg, rng: np.random.Generator) -> np.ndarray:
    """Analog phases in [0, 2 pi).

    frequency-flat: [G, Nt, Mt]; per-subcarrier: [G, N, Nt, Mt] with the MATLAB loop order
    (g outer, n inner, rand(Nt,1) innermost).
    """
    b, N, Nt = cfg.beam, cfg.physics.num_subcarriers, cfg.array.num_tx
    if b.analog_frequency_flat:
        return 2 * np.pi * rng.random((b.num_beams, Nt, b.num_streams))
    return 2 * np.pi * rng.random((b.num_beams, N, Nt, b.num_streams))


def beams_from_phases(cfg, phases) -> np.ndarray:
    """Map analog phases to F [N, G, Nt, Mt]."""
    b, N, Nt = cfg.beam, cfg.physics.num_subcarriers, cfg.array.num_tx
    phases = np.asarray(phases, dtype=float)
    F = np.exp(1j * phases)
    if b.analog_frequency_flat:
        F = np.broadcast_to(F[None], (N, b.num_beams, Nt, b.num_streams)).copy()
    else:
        F = np.transpose(F, (1, 0, 2, 3)).copy()  # [G, N, ...] -> [N, G, ...]
    if b.normalize:
        F /= np.sqrt(Nt * b.num_streams)  # ||F^{(g)}[n]||_F = 1
    return F


def generate_tx_beams(cfg, rng: np.random.Generator, phases=None) -> np.ndarray:
    if phases is None:
        phases = generate_beam_phases(cfg, rng)
    return beams_from_phases(cfg, phases)


def generate_symbols(cfg, rng: np.random.Generator | None, phases=None) -> np.ndarray:
    """x [N, G, Mt]: unit-modulus random-phase pilots (paper) or ones (GitHub)."""
    b, N = cfg.beam, cfg.physics.num_subcarriers
    shape = (N, b.num_beams, b.num_streams)
    if b.symbols == "ones":
        return np.ones(shape, dtype=complex)
    if b.symbols != "random_phase":
        raise ValueError(f"unknown symbols mode {b.symbols!r}")
    if phases is None:
        phases = 2 * np.pi * rng.random(shape)
    return np.exp(1j * np.asarray(phases, dtype=float))


def effective_beams(F: np.ndarray, x: np.ndarray) -> np.ndarray:
    """s^{(g)}[n] = F^{(g)}[n] x^{(g)}[n], shape [N, G, Nt]."""
    return np.einsum("ngtm,ngm->ngt", F, x)

"""Draw the scenario (p, alpha, {s_k}) from the configured priors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    bs_position: np.ndarray  # [2]
    ue_position: np.ndarray  # [2]
    orientation: float
    scatterers: np.ndarray  # [S, 2]
    condition: str  # los | nlos | olos

    @property
    def include_los(self) -> bool:
        return self.condition in ("los", "nlos")


def _draw_vector(s, rng, dim=2):
    if s.mode == "fixed":
        return np.asarray(s.value, dtype=float).reshape(dim)
    if s.mode == "uniform_box":
        lo, hi = np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)
        return lo + (hi - lo) * rng.random(dim)
    raise ValueError(f"unsupported sampler mode {s.mode!r} for a position")


def _draw_scalar(s, rng):
    if s.mode == "fixed":
        return float(s.value)
    if s.mode == "uniform":
        return float(s.low + (s.high - s.low) * rng.random())
    raise ValueError(f"unsupported sampler mode {s.mode!r} for a scalar")


def _draw_scatterers(s, rng):
    if s.mode == "none":
        return np.zeros((0, 2))
    if s.mode == "fixed":
        return np.asarray(s.positions, dtype=float).reshape(-1, 2)
    if s.mode == "uniform_box":
        lo, hi = np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)
        num = s.num if s.num_max is None else int(rng.integers(s.num, s.num_max + 1))
        # column-major fill to mirror MATLAB's SP = rand(L-1,2)*20-10
        u = rng.random(2 * num).reshape(2, num).T
        return lo + (hi - lo) * u
    raise ValueError(f"unsupported sampler mode {s.mode!r} for scatterers")


def _draw_scatterers_half_plane(s, rng, p, alpha):
    """Uniform in the box, restricted by rejection to (s - p) . u(alpha) < 0 (behind the MS array, F27)."""
    if s.mode != "uniform_box" or s.half_plane != "back":
        raise ValueError("half_plane='back' requires mode='uniform_box'")
    lo, hi = np.asarray(s.low, dtype=float), np.asarray(s.high, dtype=float)
    num = s.num if s.num_max is None else int(rng.integers(s.num, s.num_max + 1))
    normal = np.array([np.cos(alpha), np.sin(alpha)])
    out = []
    for _ in range(num):
        for _attempt in range(10_000):
            cand = lo + (hi - lo) * rng.random(2)
            if np.dot(cand - p, normal) < 0:
                out.append(cand)
                break
        else:
            raise RuntimeError("half-plane rejection sampling failed: the box lies in front of the MS array")
    return np.array(out).reshape(-1, 2)


def sample_scenario(cfg, rng: np.random.Generator, injected: dict | None = None) -> Scenario:
    """Draw order: scatterers, ue_position, orientation (with half_plane: ue_position, orientation, scatterers).
    ``injected`` overrides any of them."""
    sc = cfg.scenario
    inj = injected or {}
    if sc.scatterers.half_plane is None:
        scat = inj["scatterers"] if "scatterers" in inj else _draw_scatterers(sc.scatterers, rng)
        p = inj["ue_position"] if "ue_position" in inj else _draw_vector(sc.ue_position, rng)
        alpha = inj["orientation"] if "orientation" in inj else _draw_scalar(sc.orientation, rng)
    else:
        p = inj["ue_position"] if "ue_position" in inj else _draw_vector(sc.ue_position, rng)
        alpha = inj["orientation"] if "orientation" in inj else _draw_scalar(sc.orientation, rng)
        scat = inj["scatterers"] if "scatterers" in inj else \
            _draw_scatterers_half_plane(sc.scatterers, rng, np.asarray(p, float), float(alpha))
    if sc.condition not in ("los", "nlos", "olos"):
        raise ValueError(f"unknown condition {sc.condition!r}")
    scat = np.asarray(scat, dtype=float).reshape(-1, 2)
    if sc.condition == "los":
        scat = np.zeros((0, 2))
    return Scenario(np.asarray(sc.bs_position, dtype=float), np.asarray(p, dtype=float), float(alpha),
                    scat, sc.condition)

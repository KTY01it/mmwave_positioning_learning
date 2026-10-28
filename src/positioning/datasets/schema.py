"""Fixed per-sample schema (master plan P3). H is never stored: it is regenerated from the seeds.

Variable-size quantities are zero-padded to K_MAX / S_MAX with boolean masks.
"""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 1
K_MAX = 8
S_MAX = 8
CONDITIONS = ("los", "nlos", "olos")

FIELDS = {
    # observation (the only estimator input)
    "y": "complex64[N,G,Nr]",
    # pose
    "position": "float64[2]",
    "orientation": "float64",
    "condition": "int8 (0 los, 1 nlos, 2 olos)",
    "num_paths": "int16",
    # paths (padded to K_MAX, LOS first when present)
    "path_mask": "bool[K_MAX]",
    "delay": "float64[K_MAX] s",
    "aod": "float64[K_MAX] rad",
    "aoa": "float64[K_MAX] rad",
    "gain_tilde": "complex128[K_MAX]",
    "is_los": "bool[K_MAX]",
    "scatterers": "float64[S_MAX,2]",
    "scatterer_mask": "bool[S_MAX]",
    # noise / seeds
    "snr_db": "float64 (Eq. 68)",
    "n0": "float64",
    "seed": "int64",
    "seed_geometry": "int64",
    "seed_gain": "int64",
    "seed_beam": "int64",
    "seed_noise": "int64",
    "component": "int16 (index of the split component that produced the sample)",
    # bounds at the true parameters (optional, NaN if not computed)
    "peb": "float64",
    "reb": "float64",
    "crb_delay": "float64[K_MAX]",
    "crb_aod": "float64[K_MAX]",
    "crb_aoa": "float64[K_MAX]",
}


def _pad(v, n, fill=0.0, dtype=float):
    out = np.full(n, fill, dtype=dtype)
    v = np.asarray(v)
    out[: v.size] = v
    return out


def sample_to_record(sample, component: int = 0, bounds=None) -> dict:
    gt, meta, p = sample.gt, sample.meta, sample.gt.paths
    K, S = p.num_paths, gt.scatterers.shape[0]
    if K > K_MAX or S > S_MAX:
        raise ValueError(f"K={K} or S={S} exceeds the schema limits")
    scat = np.zeros((S_MAX, 2))
    scat[:S] = gt.scatterers
    nan = np.full(K_MAX, np.nan)
    return {
        "y": sample.y.astype(np.complex64),
        "position": np.asarray(gt.position, float),
        "orientation": float(gt.orientation),
        "condition": np.int8(CONDITIONS.index(gt.condition)),
        "num_paths": np.int16(K),
        "path_mask": _pad(np.ones(K, bool), K_MAX, False, bool),
        "delay": _pad(p.delay, K_MAX),
        "aod": _pad(p.aod, K_MAX),
        "aoa": _pad(p.aoa, K_MAX),
        "gain_tilde": _pad(p.gain_tilde, K_MAX, 0, complex),
        "is_los": _pad(p.is_los, K_MAX, False, bool),
        "scatterers": scat,
        "scatterer_mask": _pad(np.ones(S, bool), S_MAX, False, bool),
        "snr_db": float(meta.snr_db),
        "n0": float(meta.n0),
        "seed": np.int64(-1 if meta.seed is None else meta.seed),
        "seed_geometry": np.int64(meta.seed_geometry),
        "seed_gain": np.int64(meta.seed_gain),
        "seed_beam": np.int64(meta.seed_beam),
        "seed_noise": np.int64(meta.seed_noise),
        "component": np.int16(component),
        "peb": float("nan") if bounds is None else bounds.peb,
        "reb": float("nan") if bounds is None else bounds.reb,
        "crb_delay": nan.copy() if bounds is None else _pad(bounds.crb_delay, K_MAX, np.nan),
        "crb_aod": nan.copy() if bounds is None else _pad(bounds.crb_aod, K_MAX, np.nan),
        "crb_aoa": nan.copy() if bounds is None else _pad(bounds.crb_aoa, K_MAX, np.nan),
    }


def stack_records(records: list[dict]) -> dict:
    return {k: np.stack([r[k] for r in records]) for k in records[0]}

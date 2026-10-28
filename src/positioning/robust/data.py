"""Fresh development / confirmation samples for the robustness completion program.

Seed namespace icc2027/robust-completion/v1/<split>: SHA-256 -> SeedSequence, 64-bit streams with the top bit set.
Prior: the components of `dev_v2_iid` (configs/datasets/p3_v2.yaml) plus explicit overrides per shift. No stored split is
read. Condition = index mod 3 unless given; SNR from the sample's auxiliary stream.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "configs" / "datasets" / "p3_v2.yaml"
NAMESPACE = "icc2027/robust-completion/v1"
CONDITIONS = ("los", "nlos", "olos")
_COMPS = {}


def seeds(split: str, index: int) -> dict:
    ent = int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{split}".encode()).digest(), "big")
    st = np.random.SeedSequence(entropy=ent, spawn_key=(int(index),)).generate_state(5, np.uint64)
    return {k: int(v) | (1 << 63) for k, v in zip(("geometry", "gain", "beam", "noise", "aux"), st)}


def _components():
    if not _COMPS:
        from ..datasets.splits import load_splits
        cfg_path, splits = load_splits(SPEC)
        _COMPS["path"] = cfg_path
        _COMPS["comps"] = next(s for s in splits if s.name == "dev_v2_iid").components
    return _COMPS["path"], _COMPS["comps"]


def draw(split: str, index: int, *, snr=("uniform", -10.0, 10.0), overrides: dict | None = None,
         condition: str | None = None, keep_noiseless: bool = False):
    """One sample (32 random beams). snr: float | ("uniform", lo, hi) | ("choice", [values]).
    Returns (SimulationOutput, seeds)."""
    from ..datasets.generator import _system
    cfg_path, comps = _components()
    comp = int(index) % 3 if condition is None else CONDITIONS.index(condition)
    sd = seeds(split, index)
    rng = np.random.default_rng(sd["aux"])
    u = rng.random()
    if isinstance(snr, (int, float)):
        snr_db = float(snr)
    elif snr[0] == "uniform":
        snr_db = float(snr[1] + (snr[2] - snr[1]) * u)
    else:
        snr_db = float(snr[1][int(u * len(snr[1])) % len(snr[1])])
    ov = dict(comps[comp].overrides)
    if overrides:
        ov.update(overrides)
    s = _system(cfg_path, ov).simulate(seed=None, snr_db=snr_db, seed_geometry=sd["geometry"], seed_gain=sd["gain"],
                                       seed_beam=sd["beam"], seed_noise=sd["noise"], keep_noiseless=keep_noiseless)
    return s, sd

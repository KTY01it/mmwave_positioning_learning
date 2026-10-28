"""Fresh samples of the B2f3-Robust program. Seed namespace icc2027/b2f3-robust/v1/<split> (SHA-256 -> SeedSequence,
64-bit streams with the top bit set). Prior: components of `dev_v2_iid` (configs/datasets/p3_v2.yaml) plus explicit
overrides. No stored split is read. Condition = index mod 3 unless given; SNR from the sample's auxiliary stream."""
from __future__ import annotations

import hashlib

import numpy as np

from ..robust import data as _RD

NAMESPACE = "icc2027/b2f3-robust/v1"
CONDITIONS = _RD.CONDITIONS
BOX = lambda lo, hi: {"scenario.ue_position": {"mode": "uniform_box", "low": list(lo), "high": list(hi)}}
REGIONS = {
    "iid": {},                                   # training region x in [1.5, 5.0], y in [-0.6, 0.9]
    "lat": BOX((2.0, 1.2), (5.0, 2.0)),          # lateral box of the R3 failure (cell A2), new seeds
    "latneg": BOX((2.0, -2.0), (5.0, -1.3)),     # mirrored lateral box (never used before)
    "far": BOX((5.5, -1.2), (7.0, -0.7)),        # development OOD box of the v2 protocol, new seeds
    "farlat": BOX((5.5, 1.2), (7.0, 2.0)),       # far and lateral
}


def region_overrides(region: str, index: int) -> dict:
    """Overrides of a named region. "ori": |alpha| in [0.5, 0.8] rad (outside the declared prior), sign alternating."""
    if region == "ori":
        lo, hi = (0.5, 0.8) if (int(index) // 3) % 2 == 0 else (-0.8, -0.5)
        return {"scenario.orientation": {"mode": "uniform", "low": lo, "high": hi}}
    return REGIONS[region]


def seeds(split: str, index: int) -> dict:
    ent = int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{split}".encode()).digest(), "big")
    st = np.random.SeedSequence(entropy=ent, spawn_key=(int(index),)).generate_state(5, np.uint64)
    return {k: int(v) | (1 << 63) for k, v in zip(("geometry", "gain", "beam", "noise", "aux"), st)}


def draw(split: str, index: int, *, snr=("uniform", -10.0, 10.0), overrides: dict | None = None,
         condition: str | None = None, keep_noiseless: bool = False):
    """One sample (32 random beams). snr: float | ("uniform", lo, hi) | ("choice", [values]).
    Returns (SimulationOutput, seeds), or (None, seeds) when the scatterer geometry is infeasible for the pose."""
    from ..datasets.generator import _system
    cfg_path, comps = _RD._components()
    comp = int(index) % 3 if condition is None else CONDITIONS.index(condition)
    sd = seeds(split, index)
    u = np.random.default_rng(sd["aux"]).random()
    if isinstance(snr, (int, float)):
        snr_db = float(snr)
    elif snr[0] == "uniform":
        snr_db = float(snr[1] + (snr[2] - snr[1]) * u)
    else:
        snr_db = float(snr[1][int(u * len(snr[1])) % len(snr[1])])
    ov = dict(comps[comp].overrides)
    if overrides:
        ov.update(overrides)
    try:
        s = _system(cfg_path, ov).simulate(seed=None, snr_db=snr_db, seed_geometry=sd["geometry"], seed_gain=sd["gain"],
                                           seed_beam=sd["beam"], seed_noise=sd["noise"], keep_noiseless=keep_noiseless)
    except RuntimeError:                         # half-plane rule of the forward model: no scatterer behind the MS
        return None, sd
    return s, sd

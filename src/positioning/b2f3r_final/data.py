"""Fresh samples of the finalization program: namespace icc2027/b2f3r-final/v1/<split> (fdev_*, fval_*, fint_*, fconf_*).
Regions: those of positioning.b2f3r_v2.data plus stress regions (development only):
  wide  receiver box x in [1.5, 7.0], y in [-2.0, 2.0] (enlarged)
  rev   orientation alpha in [pi - 0.5, pi + 0.5] (array reversed: violates the declared array-facing / AoA-branch assumption)
  oriw  orientation alpha uniform in [-1.2, 1.2] rad (enlarged orientation)"""
from __future__ import annotations

import hashlib

import numpy as np

from ..b2f3r import data as D1
from ..b2f3r_v2 import data as D2

NAMESPACE = "icc2027/b2f3r-final/v1"
separation = D2.separation
EXTRA = {"wide": D1.BOX((1.5, -2.0), (7.0, 2.0)),
         "rev": {"scenario.orientation": {"mode": "uniform", "low": float(np.pi - 0.5), "high": float(np.pi + 0.5)}},
         "oriw": {"scenario.orientation": {"mode": "uniform", "low": -1.2, "high": 1.2}}}


def seeds(split, index):
    ent = int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{split}".encode()).digest(), "big")
    st = np.random.SeedSequence(entropy=ent, spawn_key=(int(index),)).generate_state(5, np.uint64)
    return {k: int(v) | (1 << 63) for k, v in zip(("geometry", "gain", "beam", "noise", "aux"), st)}


def draw(split, index, **kw):
    """positioning.b2f3r_v2.data.draw with the finalization seed namespace and the extra stress regions."""
    orig_s, orig_r = D2.seeds, D2.region_overrides
    try:
        D2.seeds = seeds
        D2.region_overrides = lambda region, i: dict(EXTRA[region]) if region in EXTRA else orig_r(region, i)
        return D2.draw(split, index, **kw)
    finally:
        D2.seeds, D2.region_overrides = orig_s, orig_r

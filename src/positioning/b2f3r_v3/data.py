"""Fresh samples of B2f3R-v3: namespace icc2027/b2f3r-v3/v1/<split> (development v3dev_*, validation v3val_*,
confirmation v3conf_*). Regions and prior: positioning.b2f3r_v2.data (imported, not edited)."""
from __future__ import annotations

import hashlib

import numpy as np

from ..b2f3r_v2 import data as D2

NAMESPACE = "icc2027/b2f3r-v3/v1"


def seeds(split, index):
    ent = int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{split}".encode()).digest(), "big")
    st = np.random.SeedSequence(entropy=ent, spawn_key=(int(index),)).generate_state(5, np.uint64)
    return {k: int(v) | (1 << 63) for k, v in zip(("geometry", "gain", "beam", "noise", "aux"), st)}


def draw(split, index, **kw):
    """Same as positioning.b2f3r_v2.data.draw, with the v3 seed namespace."""
    orig = D2.seeds
    try:
        D2.seeds = seeds
        return D2.draw(split, index, **kw)
    finally:
        D2.seeds = orig


separation = D2.separation

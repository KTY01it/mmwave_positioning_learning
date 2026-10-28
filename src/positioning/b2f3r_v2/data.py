"""Fresh samples of the B2f3R-v2 program: seed namespace icc2027/b2f3r-v2/v1/<split> (never used before), prior and
regions as in positioning.b2f3r.data (imported, not edited)."""
from __future__ import annotations

import hashlib

import numpy as np

from ..b2f3r import data as D1
from ..robust import data as _RD

NAMESPACE = "icc2027/b2f3r-v2/v1"
REGIONS = dict(D1.REGIONS)
SCAT = lambda num, lo=(1.0, 0.3), hi=(3.5, 2.0), num_max=None: {"scenario.scatterers": {
    "mode": "uniform_box", "num": num, "num_max": num_max, "low": list(lo), "high": list(hi), "half_plane": "back"}}


def region_overrides(region: str, index: int) -> dict:
    """Named regions of v1 plus shift cells: scat4 (4 scatterers, OLOS/NLOS), close (scatterers confined to
    0.5 m x 0.4 m), refl5 / refl20 (mean reflection loss -5 / -20 dB)."""
    if region in REGIONS or region == "ori":
        return D1.region_overrides(region, index)
    if region == "scat4":
        return SCAT(3 if int(index) % 3 == 1 else 4)
    if region == "close":
        c = int(index) % 3
        return {} if c == 0 else SCAT(1 if c == 1 else 3, (2.0, 0.8), (2.5, 1.2), 2 if c == 1 else None)
    if region == "refl5":
        return {"channel.reflection_loss_db_mean": -5.0}
    if region == "refl20":
        return {"channel.reflection_loss_db_mean": -20.0}
    raise KeyError(region)


def seeds(split: str, index: int) -> dict:
    ent = int.from_bytes(hashlib.sha256(f"{NAMESPACE}/{split}".encode()).digest(), "big")
    st = np.random.SeedSequence(entropy=ent, spawn_key=(int(index),)).generate_state(5, np.uint64)
    return {k: int(v) | (1 << 63) for k, v in zip(("geometry", "gain", "beam", "noise", "aux"), st)}


def draw(split: str, index: int, *, snr=("uniform", -10.0, 10.0), region: str = "iid", condition: str | None = None,
         keep_noiseless: bool = False):
    """(SimulationOutput or None if the scatterer geometry is infeasible, seeds)."""
    from ..datasets.generator import _system
    cfg_path, comps = _RD._components()
    comp = int(index) % 3 if condition is None else _RD.CONDITIONS.index(condition)
    sd = seeds(split, index)
    u = np.random.default_rng(sd["aux"]).random()
    if isinstance(snr, (int, float)):
        snr_db = float(snr)
    elif snr[0] == "uniform":
        snr_db = float(snr[1] + (snr[2] - snr[1]) * u)
    else:
        snr_db = float(snr[1][int(u * len(snr[1])) % len(snr[1])])
    ov = dict(comps[comp].overrides)
    ov.update(region_overrides(region, index))
    try:
        s = _system(cfg_path, ov).simulate(seed=None, snr_db=snr_db, seed_geometry=sd["geometry"], seed_gain=sd["gain"],
                                           seed_beam=sd["beam"], seed_noise=sd["noise"], keep_noiseless=keep_noiseless)
    except RuntimeError:
        return None, sd
    return s, sd


def separation(sample) -> dict:
    """Smallest pairwise separation of the true paths: delay in samples, sin(AoA) and sin(AoD) in array cells."""
    gp = sample.gt.paths
    cfg = sample.context.config
    if gp.num_paths < 2:
        return {"d_tau": None, "d_aoa": None, "d_aod": None}
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    iu = np.triu_indices(gp.num_paths, 1)
    dt = np.abs(np.subtract.outer(gp.delay, gp.delay))[iu] / Ts
    da = np.abs(np.subtract.outer(np.sin(gp.aoa), np.sin(gp.aoa)))[iu] * nr / 2
    dd = np.abs(np.subtract.outer(np.sin(gp.aod), np.sin(gp.aod)))[iu] * nt / 2
    return {"d_tau": float(dt.min()), "d_aoa": float(da.min()), "d_aod": float(dd.min()),
            "d_joint": float(np.min(np.maximum(dt, np.maximum(da, dd))))}

"""Dataset generation = a loop over seeds around MmWavePositioningSystem.simulate. No physics here."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..bounds.peb_reb import compute_bounds
from ..config import load_config
from ..system.simulator import MmWavePositioningSystem
from .schema import SCHEMA_VERSION, sample_to_record, stack_records
from .splits import SplitSpec

_CACHE: dict = {}


def _system(config_path: str, overrides: dict) -> MmWavePositioningSystem:
    key = (config_path, json.dumps(overrides, sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = MmWavePositioningSystem(load_config(config_path, **overrides))
    return _CACHE[key]


def simulate_split_sample(config_path: str, split: SplitSpec, seed: int, with_bounds: bool = True):
    comp, snr = split.draw(seed)
    sysm = _system(config_path, split.components[comp].overrides)
    sample = sysm.simulate(seed=int(seed), snr_db=snr, **split.fixed_seeds)
    bounds = None
    if with_bounds:
        try:
            bounds = compute_bounds(sample)
        except np.linalg.LinAlgError:  # e.g. OLOS with < 3 scatterers: singular position FIM
            bounds = None
    return sample, comp, bounds


def _record(args):
    config_path, split, seed, with_bounds = args
    sample, comp, bounds = simulate_split_sample(config_path, split, seed, with_bounds)
    return sample_to_record(sample, comp, bounds)


def generate_split(config_path: str, split: SplitSpec, out_dir: str | Path, shard_size: int = 1000,
                   with_bounds: bool = True, workers: int | None = None, spec_path: str | None = None,
                   dataset_version: int | None = None) -> Path:
    out = Path(out_dir) / split.name
    out.mkdir(parents=True, exist_ok=True)
    seeds = split.seeds()
    shards = []
    with ProcessPoolExecutor(workers) as ex:
        for i in range(0, seeds.size, shard_size):
            chunk = seeds[i:i + shard_size]
            recs = list(ex.map(_record, [(config_path, split, int(s), with_bounds) for s in chunk], chunksize=8))
            fname = f"shard_{i // shard_size:05d}.npz"
            np.savez_compressed(out / fname, **stack_records(recs))
            shards.append({"file": fname, "num_samples": len(recs), "first_seed": int(chunk[0])})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "spec_path": spec_path,
        "beams_stored": "all G of the base config; evaluate at G by taking the first G beams (nested subsets)",
        "split": {**asdict(split), "components": [asdict(c) for c in split.components]},
        "base_config_path": config_path,
        "base_config": load_config(config_path).to_dict(),
        "shards": shards,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def load_split(path: str | Path) -> dict:
    """Concatenate all shards of a stored split."""
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text())
    parts = [dict(np.load(path / s["file"])) for s in man["shards"]]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def regenerate_sample(config_path: str, split: SplitSpec, seed: int):
    """Exact SimulationOutput (incl. H, F, x) of a stored record, from its seed."""
    return simulate_split_sample(config_path, split, seed, with_bounds=False)[0]

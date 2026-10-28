"""Pre-defined IID/OOD splits (master plan P3) with disjoint seed ranges.

A split is a list of components (config overrides + weight); each sample picks one component and its SNR
deterministically from its seed, so a split is fully reproducible from (base config, split spec, seed range).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SEED_STRIDE = 10_000_000  # seed range of split i: [i*SEED_STRIDE, (i+1)*SEED_STRIDE)


@dataclass
class SplitComponent:
    overrides: dict = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class SplitSpec:
    name: str
    index: int  # determines the seed range
    num_samples: int
    components: list = field(default_factory=lambda: [SplitComponent()])
    snr: dict = field(default_factory=lambda: {"mode": "fixed", "value": 0.0})
    fixed_seeds: dict = field(default_factory=dict)  # e.g. {"seed_geometry": 1, "seed_gain": 1, "seed_beam": 1}
    stored: bool = True  # train is generated on the fly
    description: str = ""

    def seeds(self) -> np.ndarray:
        return self.index * SEED_STRIDE + np.arange(self.num_samples)

    def draw(self, seed: int):
        """(component index, snr_db) for one sample — independent stream from the simulator's."""
        rng = np.random.default_rng([int(seed), 0x5EED])
        w = np.array([c.weight for c in self.components], float)
        comp = int(rng.choice(len(w), p=w / w.sum()))
        m = self.snr.get("mode", "fixed")
        if m == "fixed":
            snr = float(self.snr["value"])
        elif m == "uniform":
            snr = float(rng.uniform(self.snr["low"], self.snr["high"]))
        elif m == "choice":
            snr = float(rng.choice(self.snr["values"]))
        else:
            raise ValueError(f"unknown snr mode {m!r}")
        return comp, snr


def load_splits(path: str | Path) -> tuple[str, list[SplitSpec]]:
    """YAML: {base_config: <path>, splits: [{name, index, num_samples, components, snr, ...}]}"""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    out = []
    for s in raw["splits"]:
        comps = [SplitComponent(c.get("overrides", {}), c.get("weight", 1.0)) for c in s.get("components", [{}])]
        out.append(SplitSpec(name=s["name"], index=s["index"], num_samples=s["num_samples"], components=comps,
                             snr=s.get("snr", {"mode": "fixed", "value": 0.0}), fixed_seeds=s.get("fixed_seeds", {}),
                             stored=s.get("stored", True), description=s.get("description", "")))
    idx = [s.index for s in out]
    if len(set(idx)) != len(idx):
        raise ValueError("split indices (seed ranges) must be unique")
    return str((path.parent / raw["base_config"]).resolve()), out

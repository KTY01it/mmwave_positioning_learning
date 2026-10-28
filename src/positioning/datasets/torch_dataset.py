"""PyTorch wrappers (optional dependency). Inputs contain ONLY y; labels are returned separately."""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("positioning.datasets.torch_dataset requires torch") from exc

from .generator import load_split, simulate_split_sample
from .schema import sample_to_record

LABEL_KEYS = ("position", "orientation", "condition", "num_paths", "path_mask", "delay", "aod", "aoa", "snr_db",
              "peb", "reb")


def _to_tensors(rec: dict):
    y = np.asarray(rec["y"])
    x = torch.from_numpy(np.stack([y.real, y.imag]).astype(np.float32))  # [2, N, G, Nr]
    labels = {k: torch.as_tensor(np.asarray(rec[k])) for k in LABEL_KEYS}
    return x, labels


class OnTheFlyDataset(Dataset):
    """Training split generated from seeds at access time (master plan P3: H and train y are not stored)."""

    def __init__(self, config_path: str, split, with_bounds: bool = False):
        self.config_path, self.split, self.with_bounds = config_path, split, with_bounds
        self.seeds = split.seeds()

    def __len__(self):
        return self.seeds.size

    def __getitem__(self, i):
        sample, comp, bounds = simulate_split_sample(self.config_path, self.split, int(self.seeds[i]), self.with_bounds)
        return _to_tensors(sample_to_record(sample, comp, bounds))


class StoredDataset(Dataset):
    def __init__(self, path):
        self.data = load_split(path)

    def __len__(self):
        return self.data["y"].shape[0]

    def __getitem__(self, i):
        return _to_tensors({k: v[i] for k, v in self.data.items()})

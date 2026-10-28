"""On-the-fly training data from the P3 split specification (no physics here: simulate(seed) + features)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..datasets.generator import simulate_split_sample
from ..evaluation.benchmark import G_GRID
from ..system.simulator import subsample_beams
from .features import angle_delay_image, encode_pose


class PoseFeatureDataset(Dataset):
    """(angle-delay image, log2(G/32), encoded pose) for seeds of a split; G drawn from G_GRID per sample."""

    def __init__(self, spec_path: str, split, seeds=None, G_choices=G_GRID, fixed_G: int | None = None):
        self.spec_path, self.split = spec_path, split
        self.seeds = split.seeds() if seeds is None else np.asarray(seeds)
        self.G_choices, self.fixed_G = tuple(G_choices), fixed_G

    def __len__(self):
        return int(self.seeds.size)

    def __getitem__(self, i):
        seed = int(self.seeds[i])
        s, _, _ = simulate_split_sample(self.spec_path, self.split, seed, with_bounds=False)
        G = self.fixed_G or int(np.random.default_rng([seed, 0x6]).choice(self.G_choices))
        s = subsample_beams(s, G)
        x = torch.from_numpy(angle_delay_image(s.y, s.context))
        g = torch.tensor([np.log2(G / 32.0)], dtype=torch.float32)
        t = torch.from_numpy(encode_pose(s.gt.position, s.gt.orientation))
        return x, g, t


class SparseChannelDataset(Dataset):
    """(y/sqrt(N0), Z, U_Rx, h_check/sqrt(N0)) for the unrolled sparse estimator. All items of a batch share G:
    G = G_choices[(i // batch_size) % len(G_choices)] (use a sequential sampler; seeds are pre-shuffled)."""

    def __init__(self, spec_path: str, split, seeds, batch_size: int, G_choices=G_GRID):
        self.spec_path, self.split = spec_path, split
        self.seeds = np.random.default_rng(1).permutation(np.asarray(seeds))
        self.bs, self.G_choices = batch_size, tuple(G_choices)

    def __len__(self):
        return int(self.seeds.size)

    def __getitem__(self, i):
        from ..signal_processing.beamspace import beamspace_channel
        from .lamp import operator_inputs, unitary_dictionaries
        s, _, _ = simulate_split_sample(self.spec_path, self.split, int(self.seeds[i]), with_bounds=False)
        s = subsample_beams(s, self.G_choices[(i // self.bs) % len(self.G_choices)])
        yn, Z, Ur = operator_inputs(s.y, s.context)
        Ut, Ur_, _, _ = unitary_dictionaries(s.context.config)
        hc = np.transpose(beamspace_channel(s.h, Ut, Ur_), (0, 2, 1)) / np.sqrt(s.context.n0)  # [N, Nbt, Nbr]
        return torch.from_numpy(yn), torch.from_numpy(Z), torch.from_numpy(Ur), torch.from_numpy(hc.astype(np.complex64))


class PathHeatmapDataset(Dataset):
    """(network input [D+1, Nt, Nr], heatmap, regression maps, mask) for the learned path initializer (6A)."""

    def __init__(self, spec_path: str, split, seeds, G_choices=G_GRID, fixed_G: int | None = None):
        self.spec_path, self.split = spec_path, split
        self.seeds = np.asarray(seeds)
        self.G_choices, self.fixed_G = tuple(G_choices), fixed_G

    def __len__(self):
        return int(self.seeds.size)

    def __getitem__(self, i):
        from .path_init import make_targets, network_input
        seed = int(self.seeds[i])
        s, _, _ = simulate_split_sample(self.spec_path, self.split, seed, with_bounds=False)
        G = self.fixed_G or int(np.random.default_rng([seed, 0x7]).choice(self.G_choices))
        s = subsample_beams(s, G)
        heat, reg, mask = make_targets(s.gt.paths, s.context.config, s.context.config.physics.sampling_period)
        return (torch.from_numpy(network_input(s.y, s.context)), torch.from_numpy(heat), torch.from_numpy(reg),
                torch.from_numpy(mask))


class ConditionDataset(Dataset):
    """(path-init network input [D+1, Nt, Nr], condition label 0/1/2 = LOS/NLOS/OLOS) for the P8 router."""

    LABELS = {"los": 0, "nlos": 1, "olos": 2}

    def __init__(self, spec_path: str, split, seeds, G_choices=G_GRID):
        self.spec_path, self.split = spec_path, split
        self.seeds = np.asarray(seeds)
        self.G_choices = tuple(G_choices)

    def __len__(self):
        return int(self.seeds.size)

    def __getitem__(self, i):
        from .path_init import network_input
        seed = int(self.seeds[i])
        s, _, _ = simulate_split_sample(self.spec_path, self.split, seed, with_bounds=False)
        s = subsample_beams(s, int(np.random.default_rng([seed, 0x8]).choice(self.G_choices)))
        return torch.from_numpy(network_input(s.y, s.context)), torch.tensor(self.LABELS[s.gt.condition])

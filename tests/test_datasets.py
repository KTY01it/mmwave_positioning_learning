"""Datasets are thin wrappers around simulate(seed): reproducible, disjoint, schema-complete."""
import numpy as np
import pytest

from positioning.datasets.generator import generate_split, load_split, regenerate_sample
from positioning.datasets.schema import FIELDS
from positioning.datasets.splits import load_splits
from conftest import CONFIGS

SPEC = CONFIGS / "datasets" / "p3_default.yaml"


def test_seed_ranges_disjoint():
    _, splits = load_splits(SPEC)
    ranges = [(s.seeds()[0], s.seeds()[-1]) for s in splits]
    for i, a in enumerate(ranges):
        for b in ranges[i + 1:]:
            assert a[1] < b[0] or b[1] < a[0]


def test_generate_and_regenerate(tmp_path):
    cfg_path, splits = load_splits(SPEC)
    sp = next(s for s in splits if s.name == "test_iid")
    sp.num_samples = 6
    path = generate_split(cfg_path, sp, tmp_path, shard_size=4, with_bounds=True, workers=2)
    d = load_split(path)
    assert set(FIELDS) <= set(d)
    assert d["y"].shape[0] == 6 and d["y"].dtype == np.complex64
    i = 3
    s = regenerate_sample(cfg_path, sp, int(d["seed"][i]))
    np.testing.assert_allclose(s.y.astype(np.complex64), d["y"][i])
    np.testing.assert_allclose(s.gt.position, d["position"][i])
    K = int(d["num_paths"][i])
    np.testing.assert_allclose(s.gt.paths.delay, d["delay"][i][:K])
    assert d["path_mask"][i].sum() == K
    assert np.all(np.isfinite(d["peb"][d["condition"] < 2]))


def test_fixed_geometry_split_varies_only_noise():
    cfg_path, splits = load_splits(SPEC)
    sp = next(s for s in splits if s.name == "test_fixed_geom")
    a = regenerate_sample(cfg_path, sp, int(sp.seeds()[0]))
    b = regenerate_sample(cfg_path, sp, int(sp.seeds()[1]))
    np.testing.assert_array_equal(a.gt.position, b.gt.position)
    np.testing.assert_array_equal(a.context.F, b.context.F)
    assert a.meta.seed_noise != b.meta.seed_noise


def test_split_draw_is_deterministic():
    _, splits = load_splits(SPEC)
    sp = splits[2]
    assert sp.draw(123) == sp.draw(123)

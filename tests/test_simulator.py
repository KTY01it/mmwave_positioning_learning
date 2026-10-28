"""Determinism and seed separation (Checkpoint P0: same seed -> identical output)."""
import numpy as np

from positioning.config import load_config
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


def test_same_seed_bit_exact(paper_cfg):
    sysm = MmWavePositioningSystem(paper_cfg)
    a, b = sysm.simulate(seed=42), sysm.simulate(seed=42)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.h, b.h)
    assert a.meta.seed_noise == b.meta.seed_noise


def test_component_seeds_are_independent():
    cfg = load_config(CONFIGS / "experiments" / "los_area.yaml")
    sysm = MmWavePositioningSystem(cfg)
    a = sysm.simulate(seed=1)
    b = sysm.simulate(seed=2, seed_geometry=a.meta.seed_geometry, seed_gain=a.meta.seed_gain,
                      seed_beam=a.meta.seed_beam)
    np.testing.assert_array_equal(a.gt.position, b.gt.position)
    np.testing.assert_array_equal(a.h, b.h)
    np.testing.assert_array_equal(a.context.F, b.context.F)
    assert not np.allclose(a.y, b.y)  # fresh noise
    c = sysm.simulate(seed=3)
    assert not np.allclose(a.gt.position, c.gt.position)
    assert 2 <= c.gt.position[0] <= 4 and 0 <= c.gt.position[1] <= 0.3


def test_ground_truth_schema(paper_cfg):
    cfg = load_config(CONFIGS / "experiments" / "olos.yaml")
    s = MmWavePositioningSystem(cfg).simulate(seed=0)
    assert s.gt.condition == "olos" and s.gt.num_paths == 3 and not s.gt.paths.is_los.any()
    assert s.gt.scatterers.shape == (3, 2)
    for k in ("snr_db", "seed_geometry", "seed_beam", "seed_noise", "seed_gain", "n0"):
        assert hasattr(s.meta, k)
    assert s.gt.paths.as_eta().shape == (15,)


def test_subsample_beams_is_nested_and_paired(paper_cfg):
    from positioning.system.simulator import subsample_beams
    s = MmWavePositioningSystem(paper_cfg).simulate(seed=5, keep_noiseless=True)
    s8 = subsample_beams(s, 8)
    assert s8.y.shape[1] == 8 and s8.context.F.shape[1] == 8 and s8.context.config.beam.num_beams == 8
    np.testing.assert_array_equal(s8.y, s.y[:, :8])
    np.testing.assert_array_equal(s8.h, s.h)
    assert s8.context.n0 == s.context.n0
    np.testing.assert_allclose(s8.noiseless, np.einsum("nrt,ngt->ngr", s8.h, s8.context.s))

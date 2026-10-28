"""SAGE refinement (Eqs. 55-60)."""
import numpy as np
import pytest

from positioning.bounds.peb_reb import compute_bounds
from positioning.config import load_config
from positioning.signal_processing.coarse import coarse_estimate
from positioning.signal_processing.model import ChannelEstimate
from positioning.signal_processing.sage import maximize_1d, sage
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


def test_maximize_1d():
    f = lambda x: -(np.asarray(x) - 0.123) ** 2
    assert maximize_1d(f, 0.1, 0.05) == pytest.approx(0.123, abs=1e-9)


def test_sage_noiseless_recovers_truth_from_perturbed_init():
    cfg = load_config(CONFIGS / "experiments" / "nlos.yaml")
    s = MmWavePositioningSystem(cfg).simulate(seed=0, snr_db=200)
    tr = ChannelEstimate.from_paths(s.gt.paths)
    init = ChannelEstimate(tr.delay + 2e-9, tr.aod + 0.01, tr.aoa - 0.01, tr.gain_tilde * 0.8)
    res = sage(s.y, s.context, init, max_iter=30, tol=1e-9)
    np.testing.assert_allclose(res.estimate.delay, tr.delay, atol=1e-14)
    np.testing.assert_allclose(res.estimate.aod, tr.aod, atol=1e-8)
    np.testing.assert_allclose(res.estimate.aoa, tr.aoa, atol=1e-8)
    np.testing.assert_allclose(res.estimate.gain_tilde, tr.gain_tilde, rtol=1e-6)
    r = np.array(res.residual_energy)
    assert np.all(np.diff(r) <= 1e-9 * r[0])  # monotone up to round-off (coordinate ascent)


@pytest.mark.slow
def test_los_channel_rmse_near_crb_at_moderate_snr():
    cfg = load_config(CONFIGS / "experiments" / "los.yaml")
    sysm = MmWavePositioningSystem(cfg)
    err, crb = [], []
    for seed in range(96):
        s = sysm.simulate(seed=seed, snr_db=0)
        est = sage(s.y, s.context, coarse_estimate(s.y, s.context)).estimate
        assert est.num_paths == 1
        err.append(est.delay[0] - s.gt.paths.delay[0])
        crb.append(compute_bounds(s).crb_delay[0] ** 2)
    ratio_db = 10 * np.log10(np.mean(np.square(err)) / np.mean(crb))
    assert abs(ratio_db) < 1.0  # 480-sample run: 0.2 dB

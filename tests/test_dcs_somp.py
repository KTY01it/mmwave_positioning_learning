"""DCS-SOMP (Algorithm 1), threshold (41), delay/gain (53)-(54), coarse estimation."""
import numpy as np
import pytest
from scipy.special import gammaincinv

from positioning.config import load_config
from positioning.signal_processing import beamspace as B
from positioning.signal_processing import delay_estimation as D
from positioning.signal_processing.coarse import coarse_estimate
from positioning.signal_processing.dcs_somp import dcs_somp, detection_threshold
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


def test_recovers_on_grid_sparse_support(small_cfg, rng):
    s = MmWavePositioningSystem(small_cfg).simulate(seed=0)
    Ut, Ur, _, _ = B.dictionaries_from_config(small_cfg)
    op = B.SensingOperator(s.context.s, Ut, Ur)
    N = small_cfg.physics.num_subcarriers
    support = [5, 77, 140]
    h = np.zeros((N, op.num_columns), complex)
    h[:, support] = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))
    y = np.einsum("nmj,nj->nm", op.explicit(), h)
    res = dcs_somp(y, op, num_atoms=3)
    assert sorted(res.atoms) == support
    np.testing.assert_allclose(res.coefficients[np.argsort(res.atoms)], h[:, support].T, atol=1e-10)
    np.testing.assert_allclose(res.residual, 0, atol=1e-10)


def test_threshold_matches_eq41_and_false_alarm_rate(small_cfg):
    N, J = 8, 17 * 15
    d = detection_threshold(2.0, N, J, 1e-2)
    assert d == pytest.approx(2.0 * gammaincinv(N, 0.99 ** (1 / J)))
    # empirical: noise-only, unit-norm orthogonal columns -> P(max energy > delta) ~ Pfa
    rng = np.random.default_rng(1)
    trials = 4000
    e = (rng.standard_normal((trials, J, N)) ** 2 + rng.standard_normal((trials, J, N)) ** 2).sum(-1) / 2 * 2.0
    assert np.mean(e.max(1) > d) == pytest.approx(1e-2, abs=6e-3)


def test_noise_only_detects_nothing_with_high_probability(small_cfg):
    sysm = MmWavePositioningSystem(small_cfg.replace(**{"estimator.min_paths": 0}))
    false = 0
    for seed in range(20):
        s = sysm.simulate(seed=seed, snr_db=-60)  # signal negligible
        false += coarse_estimate(s.y, s.context).num_paths > 0
    assert false <= 2
    s = MmWavePositioningSystem(small_cfg).simulate(seed=0, snr_db=-60)
    assert coarse_estimate(s.y, s.context).num_paths == 1  # min_paths = 1: always one estimate


def test_delay_estimators():
    Ts, N = 1e-8, 20
    tau = 37.3e-9
    coef = 0.7j * D.delay_steering(tau, N, Ts)
    assert D.periodogram_delay(coef, Ts) == pytest.approx(tau, abs=1e-15)
    assert D.phase_slope_delay(coef, Ts) == pytest.approx(tau, abs=1e-15)
    assert D.gain_from_delay(coef, tau, 1.0, Ts) == pytest.approx(0.7j)
    # wrap-around: tau close to N Ts
    coef = D.delay_steering(199e-9, N, Ts)
    assert D.periodogram_delay(coef, Ts) == pytest.approx(199e-9, abs=1e-13)


@pytest.mark.parametrize("name,K", [("los.yaml", 1), ("nlos.yaml", 2), ("olos.yaml", 3)])
def test_coarse_estimate_detects_true_number_of_paths(name, K):
    cfg = load_config(CONFIGS / "experiments" / name)
    sysm = MmWavePositioningSystem(cfg)
    for snr in (10.0, 30.0):
        s = sysm.simulate(seed=1, snr_db=snr)
        est = coarse_estimate(s.y, s.context)
        assert est.num_paths == K, (name, snr)
        # match paths by AoD: in nlos.yaml the two delays differ by only 0.03 Ts
        e, o = est.subset(np.argsort(est.aod)), np.argsort(s.gt.paths.aod)
        np.testing.assert_allclose(e.delay, s.gt.paths.delay[o], atol=0.05 * cfg.physics.sampling_period)
        np.testing.assert_allclose(e.aod, s.gt.paths.aod[o], atol=0.01)
        np.testing.assert_allclose(np.sin(e.aoa), np.sin(s.gt.paths.aoa[o]), atol=0.01)


def test_literal_grid_dcs_somp_overestimates_K_at_high_snr():
    """F17: Algorithm 1 with grid atoms + threshold (41) over-detects an off-grid LOS path."""
    cfg = load_config(CONFIGS / "experiments" / "los.yaml", **{"estimator.detection": "dcs_somp"})
    s = MmWavePositioningSystem(cfg).simulate(seed=5, snr_db=10)
    assert coarse_estimate(s.y, s.context).num_paths > 1

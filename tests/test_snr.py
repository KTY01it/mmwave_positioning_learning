"""Level 1: SNR definition Eq. (68) (F7). Checkpoint P0: measured SNR within +-0.1 dB of target."""
import numpy as np
import pytest

from positioning.config import load_config
from positioning.signal_processing import beamspace
from positioning.system import noise
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


@pytest.mark.parametrize("snr_db", [-20.0, 0.0, 10.0])
def test_measured_snr_matches_target(snr_db):
    cfg = load_config(CONFIGS / "experiments" / "nlos.yaml")
    sysm = MmWavePositioningSystem(cfg)
    sig = noi = 0.0
    for seed in range(40):
        s = sysm.simulate(seed=seed, snr_db=snr_db, keep_noiseless=True)
        sig += np.sum(np.abs(s.noiseless) ** 2) / s.meta.n0
        noi += np.sum(np.abs(s.y - s.noiseless) ** 2) / s.meta.n0
    assert 10 * np.log10(sig / noi) == pytest.approx(snr_db, abs=0.1)


def test_eq68_in_beamspace_equals_element_domain(small_cfg):
    """||Omega h_check||^2 (beamspace, unitary U) == ||H F x||^2 (element domain)."""
    s = MmWavePositioningSystem(small_cfg).simulate(seed=1, keep_noiseless=True)
    est = small_cfg.estimator
    lam_c = small_cfg.physics.wavelength_c
    d = small_cfg.antenna_spacing
    Ut = beamspace.dictionary(small_cfg.array.num_tx, lam_c, d, small_cfg.array.index_origin, 1, "centered", True)
    Ur = beamspace.dictionary(small_cfg.array.num_rx, lam_c, d, small_cfg.array.index_origin, 1, "centered", True)
    Om = beamspace.sensing_matrix(s.context.s, Ut, Ur)
    Hc = beamspace.beamspace_channel(s.h, Ut, Ur)
    hc = beamspace.vec_beamspace(Hc)
    num = sum(np.linalg.norm(Om[n] @ hc[n]) ** 2 for n in range(Om.shape[0]))
    assert num == pytest.approx(np.sum(np.abs(s.noiseless) ** 2), rel=1e-10)
    # and Omega h_check reproduces the stacked noiseless observation exactly
    np.testing.assert_allclose(np.einsum("nij,nj->ni", Om, hc), beamspace.stack_observation(s.noiseless), atol=1e-12)


def test_n0_mode_reports_implied_snr(github_cfg):
    s = MmWavePositioningSystem(github_cfg).simulate(seed=0, keep_noiseless=True)
    assert s.meta.n0 == pytest.approx(0.01)
    assert s.meta.snr_db == pytest.approx(noise.snr_db_from_n0(s.noiseless, 0.01))

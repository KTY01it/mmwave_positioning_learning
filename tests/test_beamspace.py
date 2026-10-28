"""Level 1: beamspace Eqs. (31)-(40) incl. F4 (unitary for odd/even Nt) and F5 (peak location, Eq. 33)."""
import numpy as np
import pytest

from positioning.signal_processing import beamspace as B
from positioning.system import arrays, channel
from positioning.system.simulator import MmWavePositioningSystem


@pytest.mark.parametrize("n", [64, 65, 32, 7])
@pytest.mark.parametrize("origin", ["centered", "zero"])
def test_dictionary_unitary_any_parity(n, origin):
    U = B.dictionary(n, 0.005, 0.0025, origin, 1, "centered", True)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(n), atol=1e-12)


def test_odd_grid_contains_broadside_even_does_not():
    assert 0.0 in B.dictionary_grid(65)
    assert 0.0 not in B.dictionary_grid(64)
    np.testing.assert_allclose(B.dictionary_grid(64, 2, "github"), np.arange(-64, 64) / 128)


def test_github_dictionary_matches_main_m():
    Nt, Nb = 32, 64
    aa = 2 * np.arange(-Nb / 2, Nb / 2) / Nb
    ref = np.stack([np.exp(-1j * np.pi * a * np.arange(Nt)) for a in aa], axis=1)
    np.testing.assert_allclose(B.dictionary(Nt, 1.0, 0.5, "zero", 2, "github", False), ref, atol=1e-12)


def test_beamspace_peak_at_predicted_index(paper_cfg):
    """|H_check[n]| peaks at i = round(Nr (d/lambda) sin theta_Rx) etc. (Eq. 33), unitary Nt x Nt dictionary."""
    cfg = paper_cfg.replace(**{"estimator.dictionary_oversampling": 1})
    s = MmWavePositioningSystem(cfg).simulate(seed=0)
    Ut, Ur, sin_t, sin_r = B.dictionaries_from_config(cfg)
    Hc = B.beamspace_channel(s.h, Ut, Ur)
    ir, it = np.unravel_index(np.argmax(np.abs(Hc[0])), Hc[0].shape)
    assert it == np.argmin(np.abs(sin_t - np.sin(s.gt.paths.aod[0])))
    assert ir == np.argmin(np.abs(sin_r - np.sin(s.gt.paths.aoa[0])))
    # on-grid path: all energy in one beamspace entry on the reference subcarrier
    on_grid = channel.PathParameters(np.array([10e-9]), np.array([np.arcsin(sin_t[40])]), np.array([np.pi - np.arcsin(sin_r[20])]),
                                     np.ones(1), np.ones(1) * 65, np.ones(1), np.array([True]))
    H = channel.build_channel(on_grid, fc=cfg.physics.fc, bandwidth=cfg.physics.bandwidth, num_subcarriers=1,
                              num_tx=65, num_rx=65, c=cfg.physics.speed_of_light, spacing=cfg.antenna_spacing)
    Hc1 = np.abs(B.beamspace_channel(H, Ut, Ur)[0])
    assert Hc1[20, 40] == pytest.approx(65.0) and np.sum(Hc1 > 1e-8) == 1


def test_sensing_operator_matches_explicit(small_cfg, rng):
    s = MmWavePositioningSystem(small_cfg).simulate(seed=1)
    Ut, Ur, _, _ = B.dictionaries_from_config(small_cfg)
    op = B.SensingOperator(s.context.s, Ut, Ur)
    Om = B.sensing_matrix(s.context.s, Ut, Ur)
    np.testing.assert_allclose(op.explicit(), Om)
    r = rng.standard_normal((Om.shape[0], Om.shape[1])) + 1j * rng.standard_normal((Om.shape[0], Om.shape[1]))
    np.testing.assert_allclose(op.correlate(r), np.einsum("nmj,nm->nj", Om.conj(), r), atol=1e-12)
    np.testing.assert_allclose(op.column_norms(), np.linalg.norm(Om, axis=1))
    np.testing.assert_allclose(op.columns(17), Om[:, :, 17])


def test_angle_branches():
    assert B.sine_to_angle(0.1, "front") == pytest.approx(np.arcsin(0.1))
    assert B.sine_to_angle(0.1, "back") == pytest.approx(np.pi - np.arcsin(0.1))
    assert B.sine_to_angle(-0.1, "back") == pytest.approx(-np.pi + np.arcsin(0.1))

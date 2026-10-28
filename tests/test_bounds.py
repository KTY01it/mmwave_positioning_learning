"""FIM (Eq. 69, summed over g), CRB, PEB/REB (Eqs. 15, 27, 28)."""
import numpy as np
import pytest

from positioning.bounds.crb import channel_crb, position_fim
from positioning.bounds.fim import block_diagonal, channel_fim
from positioning.bounds.peb_reb import compute_bounds, peb, reb
from positioning.config import load_config
from positioning.signal_processing.model import ChannelEstimate, ObservationModel
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


def test_jacobian_of_mean_finite_difference(small_cfg):
    cfg = load_config(CONFIGS / "experiments" / "nlos.yaml",
                      **{"array.num_tx": 9, "array.num_rx": 7, "beam.num_beams": 4, "physics.num_subcarriers": 5})
    s = MmWavePositioningSystem(cfg).simulate(seed=2, keep_noiseless=True)
    m = ObservationModel(s.context)
    est = ChannelEstimate.from_paths(s.gt.paths)
    np.testing.assert_allclose(m.mean(est), s.noiseless, atol=1e-15 + 1e-12 * np.abs(s.noiseless).max())
    D = m.jacobian(est)
    eta = est.as_eta()
    steps = np.tile([1e-13, 1e-7, 1e-7, 1e-9, 1e-9], est.num_paths)
    for i in range(eta.size):
        e1, e0 = eta.copy(), eta.copy()
        e1[i] += steps[i]; e0[i] -= steps[i]
        fd = (m.mean(ChannelEstimate.from_eta(e1)) - m.mean(ChannelEstimate.from_eta(e0))).ravel() / (2 * steps[i])
        np.testing.assert_allclose(D[:, i], fd, rtol=1e-5, atol=1e-6 * np.abs(D[:, i]).max())


def test_fim_scales_with_snr_and_beams():
    cfg = load_config(CONFIGS / "experiments" / "los.yaml")
    sysm = MmWavePositioningSystem(cfg)
    b0 = compute_bounds(sysm.simulate(seed=0, snr_db=0))
    b20 = compute_bounds(sysm.simulate(seed=0, snr_db=20))
    assert b0.peb / b20.peb == pytest.approx(10.0, rel=1e-6)
    assert b0.crb_delay[0] / b20.crb_delay[0] == pytest.approx(10.0, rel=1e-6)


def test_fim_g_sum_equals_sum_of_single_g(small_cfg):
    """F14: FIM over G transmissions == sum of the G single-transmission FIMs (Appendix A with G=1)."""
    s = MmWavePositioningSystem(small_cfg).simulate(seed=0)
    J = channel_fim(s.context, s.gt.paths)
    G = s.context.F.shape[1]
    parts = 0
    for g in range(G):
        ctx = type(s.context)(s.context.config, s.context.F[:, g:g + 1], s.context.x[:, g:g + 1], s.context.n0,
                              s.context.bs_position, s.context.wavelengths)
        parts = parts + channel_fim(ctx, s.gt.paths)
    d = np.sqrt(np.outer(np.diag(J), np.diag(J)))  # compare correlation-normalised entries
    np.testing.assert_allclose(J / d, parts / d, atol=1e-10)


def test_appendix_a_eq70_delay_entry(small_cfg):
    """Psi_n(tau, tau) of Eq. (70) for G = 1, single path, summed over n, equals the FIM entry."""
    s = MmWavePositioningSystem(small_cfg).simulate(seed=0)
    ctx = type(s.context)(s.context.config, s.context.F[:, :1], s.context.x[:, :1], s.context.n0,
                          s.context.bs_position, s.context.wavelengths)
    J = channel_fim(ctx, s.gt.paths)
    m = ObservationModel(ctx)
    p = s.gt.paths
    N, Ts = m.N, m.Ts
    tot = 0.0
    for n in range(N):
        aT = m.tx_projection(p.aod[0])[n, 0]  # a_Tx,F,n = F^H a_Tx -> conj
        aR = m.rx_response(p.aoa[0])[n]
        A2 = (2 * np.pi * n / (N * Ts)) ** 2 * abs(aT) ** 2  # A^{(2)} with x x^H, |x| = 1
        tot += 2 / s.context.n0 * np.real(abs(p.gain_tilde[0]) ** 2 * np.vdot(aR, aR) * A2)
    assert J[0, 0] == pytest.approx(tot, rel=1e-10)


def test_peb_equals_schur_complement():
    cfg = load_config(CONFIGS / "experiments" / "nlos.yaml")
    s = MmWavePositioningSystem(cfg).simulate(seed=0, snr_db=0)
    b = compute_bounds(s)
    Jp = b.J_pos
    A, Bm, C = Jp[:3, :3], Jp[:3, 3:], Jp[3:, 3:]
    efim = A - Bm @ np.linalg.solve(C, Bm.T)
    C3 = np.linalg.inv(efim)
    assert b.peb == pytest.approx(np.sqrt(C3[0, 0] + C3[1, 1]), rel=1e-6)
    assert b.reb == pytest.approx(np.sqrt(C3[2, 2]), rel=1e-6)
    # approximate EFIM (no cross-path terms, Eq. 29) is close for well separated paths
    ba = compute_bounds(s, approximate=True)
    assert ba.peb == pytest.approx(b.peb, rel=0.2)


def test_nlos_peb_not_larger_than_los_only():
    cfg_n = load_config(CONFIGS / "experiments" / "nlos.yaml")
    s = MmWavePositioningSystem(cfg_n).simulate(seed=0, snr_db=0)
    b = compute_bounds(s)
    # same realisation, LOS path only
    J_los = b.J_eta[:5, :5]
    from positioning.system import geometry
    T = geometry.transformation_matrix(s.gt.bs_position, s.gt.position, s.gt.orientation, np.zeros((0, 2)), True,
                                       cfg_n.physics.speed_of_light)
    assert b.peb <= peb(T @ J_los @ T.T) * (1 + 1e-9)


@pytest.mark.parametrize("name,has_los,K", [("los.yaml", True, 1), ("nlos.yaml", True, 2), ("olos.yaml", False, 3)])
def test_position_fim_parameterisation_per_condition(name, has_los, K):
    """Item 7: LOS eta_bar = [p, alpha, h0]; NLOS adds a scatterer nuisance per NLOS path; OLOS = [p, alpha, {s_k, h_k}]."""
    cfg = load_config(CONFIGS / "experiments" / name)
    s = MmWavePositioningSystem(cfg).simulate(seed=0, snr_db=0)
    b = compute_bounds(s)
    assert b.J_eta.shape == (5 * K, 5 * K)
    assert b.J_pos.shape[0] == (3 + 2 + 4 * (K - 1) if has_los else 3 + 4 * K)
    assert np.isfinite(b.peb) and np.isfinite(b.reb)

"""Level 1: channel Eqs. (1)-(4), path loss (66)-(67), observation (6)."""
import numpy as np
import pytest

from positioning.system import arrays, beamforming, channel, geometry, observation
from positioning.system.simulator import MmWavePositioningSystem


def _paths(K=2):
    return channel.PathParameters(
        delay=np.array([13e-9, 20e-9])[:K], aod=np.array([0.0, 0.3])[:K], aoa=np.array([3.0, 2.7])[:K],
        gain=np.array([1.0, 0.3j])[:K], gain_tilde=np.array([5.0, 1.5j])[:K], rho=np.ones(K),
        is_los=np.array([True, False])[:K])


def _kw(**over):
    kw = dict(fc=60e9, bandwidth=100e6, num_subcarriers=6, num_tx=9, num_rx=7, c=3e8, spacing=0.0025)
    kw.update(over)
    return kw


def test_single_path_is_rank_one_outer_product():
    p = _paths(1)
    H = channel.build_channel(p, **_kw())
    assert H.shape == (6, 7, 9)
    lam = arrays.subcarrier_wavelengths(3e8, 60e9, 100e6, 6)
    for n in range(6):
        aR = arrays.ula_response(3.0, 7, lam[n], 0.0025)
        aT = arrays.ula_response(0.0, 9, lam[n], 0.0025)
        ref = 5.0 * np.exp(-2j * np.pi * n * 13e-9 / (6 * 1e-8)) * np.outer(aR, aT.conj())
        np.testing.assert_allclose(H[n], ref, atol=1e-12)
        assert np.linalg.matrix_rank(H[n]) == 1


def test_multipath_is_superposition_and_matrix_form():
    p = _paths(2)
    H = channel.build_channel(p, **_kw())
    H0 = channel.build_channel(channel.PathParameters(*(np.atleast_1d(getattr(p, f.name)[:1]) for f in channel.PathParameters.__dataclass_fields__.values())), **_kw())
    H1 = channel.build_channel(channel.PathParameters(*(np.atleast_1d(getattr(p, f.name)[1:]) for f in channel.PathParameters.__dataclass_fields__.values())), **_kw())
    np.testing.assert_allclose(H, H0 + H1, atol=1e-12)
    # Eq. (1) H = A_Rx Gamma A_Tx^H with Gamma from Eq. (4) written via gain_tilde
    lam = arrays.subcarrier_wavelengths(3e8, 60e9, 100e6, 6)
    n = 3
    ARx = arrays.ula_response(p.aoa, 7, lam[n], 0.0025).T
    ATx = arrays.ula_response(p.aod, 9, lam[n], 0.0025).T
    Gam = np.diag(p.gain_tilde * np.exp(-2j * np.pi * n * p.delay / (6 * 1e-8)))
    np.testing.assert_allclose(H[n], ARx @ Gam @ ATx.conj().T, atol=1e-12)


def test_github_channel_matches_main_m_line57():
    """H += h*exp(-1j*2*pi*TOA*(n-1)/(N*Ts))*sqrt(Nr)*getResponse(Nr,sin(AOA))*sqrt(Nt)*getResponse(Nt,sin(AOD))'"""
    Nt = Nr = 8; N = 4; Ts = 1 / 100.0  # us
    TOA, AOD, AOA, h = 0.02, 0.4, 2.9, 10.0
    gr = lambda M, phi: np.exp(-1j * np.pi * phi * np.arange(M)) / np.sqrt(M)
    ref = np.stack([h * np.exp(-1j * 2 * np.pi * TOA * n / (N * Ts)) * np.sqrt(Nr) * np.outer(gr(Nr, np.sin(AOA)), (np.sqrt(Nt) * gr(Nt, np.sin(AOD))).conj()) for n in range(N)])
    p = channel.PathParameters(np.array([TOA * 1e-6]), np.array([AOD]), np.array([AOA]), np.array([h]),
                               np.array([h * np.sqrt(Nt * Nr)]), np.ones(1), np.array([True]))
    H = channel.build_channel(p, fc=60e9, bandwidth=100e6, num_subcarriers=N, num_tx=Nt, num_rx=Nr, c=3e8,
                              spacing=3e8 / 60e9 / 2, index_origin="zero", frequency_dependent=False)
    np.testing.assert_allclose(H, ref, atol=1e-10)


def test_path_loss_eq66_67():
    lam = 3e8 / 60e9
    rho = channel.path_loss([4.0, 4.5], [4.0, 2.5], [True, False], lam, sigma0_sq=[1.0, 0.1],
                            poisson_density=1 / 7, atmospheric_db_per_km=16)
    xi = lambda d: 10 ** (-16 * d / 1e4)
    assert 1 / rho[0] == pytest.approx(xi(4.0) * (lam / (4 * np.pi * 4.0)) ** 2)
    p0 = (2.5 / 7) ** 2 * np.exp(-2.5 / 7)
    assert 1 / rho[1] == pytest.approx(0.1 * p0 * xi(4.5) * (lam / (4 * np.pi * 4.5)) ** 2)


def test_beams_paper_vs_github(paper_cfg, github_cfg, rng):
    F = beamforming.generate_tx_beams(paper_cfg, rng)
    N, G, Nt = paper_cfg.physics.num_subcarriers, paper_cfg.beam.num_beams, paper_cfg.array.num_tx
    assert F.shape == (N, G, Nt, 1)
    np.testing.assert_allclose(np.linalg.norm(F[..., 0], axis=-1), 1.0)  # ||F||_F = 1
    np.testing.assert_allclose(F[0], F[-1])  # analog: frequency flat
    Fg = beamforming.generate_tx_beams(github_cfg, rng)
    np.testing.assert_allclose(np.abs(Fg), 1.0)
    assert not np.allclose(Fg[0], Fg[1])
    x = beamforming.generate_symbols(paper_cfg, rng)
    np.testing.assert_allclose(np.abs(x), 1.0)


def test_observation_eq6(small_cfg):
    s = MmWavePositioningSystem(small_cfg).simulate(seed=3, keep_noiseless=True)
    n, g = 2, 5
    clean = s.h[n] @ s.context.F[n, g] @ s.context.x[n, g]
    np.testing.assert_allclose(s.noiseless[n, g], clean, atol=1e-15)
    assert s.y.shape == (small_cfg.physics.num_subcarriers, small_cfg.beam.num_beams, small_cfg.array.num_rx)

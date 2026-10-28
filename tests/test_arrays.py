"""Level 1: frequency-dependent ULA response, Eq. (5)."""
import numpy as np
import pytest

from positioning.system import arrays as A


def test_subcarrier_wavelengths():
    lam = A.subcarrier_wavelengths(3e8, 60e9, 100e6, 20, True)
    np.testing.assert_allclose(lam, 3e8 / (60e9 + np.arange(20) * 5e6))
    np.testing.assert_allclose(A.subcarrier_wavelengths(3e8, 60e9, 100e6, 20, False), 3e8 / 60e9)


def test_unit_norm_and_shapes():
    lam = A.subcarrier_wavelengths(3e8, 60e9, 100e6, 4)
    a = A.ula_response(np.array([0.1, -0.3, 1.0]), 9, lam, 0.0025)
    assert a.shape == (3, 4, 9)
    np.testing.assert_allclose(np.linalg.norm(a, axis=-1), 1.0)
    assert A.ula_response(0.1, 9, lam, 0.0025).shape == (4, 9)
    assert A.ula_response(np.array([0.1, 0.2]), 9, 0.005, 0.0025).shape == (2, 9)


def test_matches_paper_eq5_literal():
    Nt, lam, d, th = 7, 0.005, 0.0025, 0.4
    i = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    ref = np.exp(-1j * i * 2 * np.pi / lam * d * np.sin(th)) / np.sqrt(Nt)
    np.testing.assert_allclose(A.ula_response(th, Nt, lam, d, "centered"), ref)


def test_matches_getResponse_m():
    """getResponse(N, phi) = exp(-1j*pi*phi*(0:N-1)')/sqrt(N) with phi = sin(theta), d = lambda/2."""
    N, th = 32, 0.3
    ref = np.exp(-1j * np.pi * np.sin(th) * np.arange(N)) / np.sqrt(N)
    np.testing.assert_allclose(A.ula_response(th, N, 1.0, 0.5, "zero"), ref)


def test_index_origin_changes_phase_only():
    a_c = A.ula_response(0.3, 8, 1.0, 0.5, "centered")
    a_z = A.ula_response(0.3, 8, 1.0, 0.5, "zero")
    ratio = a_c / a_z
    np.testing.assert_allclose(np.abs(ratio), 1.0)
    np.testing.assert_allclose(ratio, ratio[0])


def test_narrowband_limit():
    lam_n = A.subcarrier_wavelengths(3e8, 60e9, 100e6, 20)
    a_wide = A.ula_response(0.5, 65, lam_n, 3e8 / 60e9 / 2)
    a_narrow = A.ula_response(0.5, 65, 3e8 / 60e9, 3e8 / 60e9 / 2)
    # B/fc = 1/600: |<a_n, a_c>| close to 1 on every subcarrier
    assert np.min(np.abs(a_wide @ a_narrow.conj())) > 0.99


@pytest.mark.parametrize("origin", ["centered", "zero"])
def test_derivative_finite_difference(origin):
    lam = np.array([0.005, 0.0049])
    th, eps = np.array([0.2, -1.1]), 1e-6
    d = A.ula_response_derivative(th, 11, lam, 0.0025, origin)
    fd = (A.ula_response(th + eps, 11, lam, 0.0025, origin) - A.ula_response(th - eps, 11, lam, 0.0025, origin)) / (2 * eps)
    np.testing.assert_allclose(d, fd, atol=1e-7)

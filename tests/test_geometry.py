"""Level 1: geometry Eqs. (17)-(22), Jacobian T (Eq. 26), inverse geometry (61), (62), (65)."""
import numpy as np
import pytest

from positioning.system import geometry as G

C = 299_792_000.0


def test_wrap_angle():
    assert G.wrap_angle(np.pi) == pytest.approx(np.pi)
    assert G.wrap_angle(-np.pi) == pytest.approx(np.pi)
    assert G.wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    np.testing.assert_allclose(G.wrap_angle(np.array([0.1, 2 * np.pi + 0.1])), [0.1, 0.1])


def test_los_known_values():
    g = G.compute_los_geometry([0, 0], [4, 0], 0.1, C)
    assert g["delay"] == pytest.approx(4 / C)
    assert g["aod"] == pytest.approx(0.0)
    assert g["aoa"] == pytest.approx(np.pi - 0.1)
    g = G.compute_los_geometry([0, 0], [3, 4], 0.0, C)
    assert g["distance"] == pytest.approx(5.0)
    assert g["aod"] == pytest.approx(np.arctan2(4, 3))
    assert g["aoa"] == pytest.approx(G.wrap_angle(np.arctan2(4, 3) + np.pi))


def test_nlos_known_values():
    g = G.compute_nlos_geometry([0, 0], [4, 0], [1.5, 0.4], 0.1, C)
    d1, d2 = np.hypot(1.5, 0.4), np.hypot(2.5, 0.4)
    assert g["delay"] == pytest.approx((d1 + d2) / C)
    assert g["aod"] == pytest.approx(np.arctan2(0.4, 1.5))
    assert g["aoa"] == pytest.approx(np.arctan2(0.4, -2.5) - 0.1)


@pytest.mark.parametrize("p,s,alpha", [([4, 0.0], [1.5, 0.4], 0.1), ([3, 0.2], [1.5, 1.4], -0.3),
                                       ([5, 1.0], [2.0, 3.0], 0.2)])
def test_matches_paper_arccos_form_in_its_valid_region(p, s, alpha):
    """F1: atan2 form == Eqs. (17)-(22) whenever p_y >= q_y and s_y >= p_y."""
    ref = G.paper_geometry_arccos([0, 0], p, alpha, s, C)
    los = G.compute_los_geometry([0, 0], p, alpha, C)
    nl = G.compute_nlos_geometry([0, 0], p, s, alpha, C)
    assert los["delay"] == pytest.approx(ref["tau0"])
    assert nl["delay"] == pytest.approx(ref["tauk"])
    assert los["aod"] == pytest.approx(ref["aod0"])
    assert nl["aod"] == pytest.approx(ref["aodk"])
    assert G.wrap_angle(los["aoa"] - ref["aoa0"]) == pytest.approx(0, abs=1e-12)
    assert G.wrap_angle(nl["aoa"] - ref["aoak"]) == pytest.approx(0, abs=1e-12)


def test_arccos_form_fails_below_axis():
    """F1: the printed arccos form loses the sign for p_y < q_y; atan2 does not."""
    ref = G.paper_geometry_arccos([0, 0], [3, -1], 0.0, [1, 1], C)
    los = G.compute_los_geometry([0, 0], [3, -1], 0.0, C)
    assert los["aod"] == pytest.approx(-ref["aod0"])


def test_matches_main_m_formulas():
    """main.m L30-36 (AOA(1) differs by 2 pi only)."""
    p, alpha, s = np.array([5.0, 1.0]), 0.2, np.array([-3.0, 7.0])
    los = G.compute_los_geometry([0, 0], p, alpha, 300e6)
    assert los["aoa"] == pytest.approx(G.wrap_angle(np.arctan2(p[1], p[0]) - np.pi - alpha))
    nl = G.compute_nlos_geometry([0, 0], p, s, alpha, 300e6)
    assert nl["aoa"] == pytest.approx(G.wrap_angle(np.arctan2(s[1] - p[1], s[0] - p[0]) - alpha))
    assert nl["delay"] == pytest.approx((np.linalg.norm(s) + np.linalg.norm(p - s)) / 300e6)


def _eta_geo(bs, p, alpha, scat, has_los):
    g = G.compute_path_geometry(bs, p, alpha, scat, has_los, C)
    return np.stack([g.delay, g.aod, g.aoa], axis=1)


@pytest.mark.parametrize("has_los,scat", [(True, []), (True, [[1.5, 0.4]]),
                                          (True, [[1.5, 0.4], [1.5, 0.9]]),
                                          (False, [[1.5, 0.4], [1.5, 0.9], [1.5, 1.4]])])
def test_transformation_matrix_finite_difference(has_los, scat):
    bs, p, alpha = np.zeros(2), np.array([4.0, 0.1]), 0.1
    scat = np.asarray(scat, dtype=float).reshape(-1, 2)
    T = G.transformation_matrix(bs, p, alpha, scat, has_los, C)
    layout = G.geometry_parameter_layout(int(has_los) + len(scat), has_los)
    assert T.shape == (len(layout), 5 * (int(has_los) + len(scat)))
    eps = 1e-7
    for i, name in enumerate(layout):
        if name[0] == "h":
            continue
        dp, da, ds = np.zeros(2), 0.0, np.zeros_like(scat)
        if name == "px": dp[0] = eps
        elif name == "py": dp[1] = eps
        elif name == "alpha": da = eps
        else:
            k = int(name[1:-1]) - int(has_los)
            ds[k, 0 if name[-1] == "x" else 1] = eps
        e1 = _eta_geo(bs, p + dp, alpha + da, scat + ds, has_los)
        e0 = _eta_geo(bs, p - dp, alpha - da, scat - ds, has_los)
        fd = (e1 - e0) / (2 * eps)
        fd[:, 1:] = G.wrap_angle(e1[:, 1:] - e0[:, 1:]) / (2 * eps)
        analytic = T[i].reshape(-1, 5)[:, :3]
        np.testing.assert_allclose(analytic * [C, 1, 1], fd * [C, 1, 1], rtol=1e-5, atol=1e-6)


def test_los_inverse_roundtrip():
    p, alpha = np.array([3.2, 0.25]), -0.15
    g = G.compute_los_geometry([0, 0], p, alpha, C)
    ph, ah = G.los_position_from_parameters(g["delay"], g["aod"], g["aoa"], [0, 0], C)
    np.testing.assert_allclose(ph, p, atol=1e-12)
    assert ah == pytest.approx(alpha)


def test_scatterer_from_lines_roundtrip():
    p, alpha, s = np.array([4.0, 0.0]), 0.1, np.array([1.5, 0.4])
    g = G.compute_nlos_geometry([0, 0], p, s, alpha, C)
    np.testing.assert_allclose(G.scatterer_from_lines([0, 0], p, alpha, g["aod"], g["aoa"]), s, atol=1e-12)


def test_olos_linear_solve_roundtrip():
    """Eq. (65) with the true alpha recovers p and the scatterers (F16)."""
    p, alpha = np.array([4.0, 0.0]), 0.1
    S = np.array([[1.5, 0.4], [1.5, 0.9]])
    gs = [G.compute_nlos_geometry([0, 0], p, s, alpha, C) for s in S]
    ph, d1, sh = G.olos_linear_solve([0, 0], alpha, [g["delay"] for g in gs], [g["aod"] for g in gs],
                                     [g["aoa"] for g in gs], C)
    np.testing.assert_allclose(ph, p, atol=1e-9)
    np.testing.assert_allclose(sh, S, atol=1e-9)
    np.testing.assert_allclose(d1, np.linalg.norm(S, axis=1), atol=1e-9)

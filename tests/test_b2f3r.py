"""B2f3-Robust candidate namespace: the default configuration is the frozen estimator; components are observation-only."""
import dataclasses

import numpy as np
import pytest

from positioning.b2f3r import data as D
from positioning.b2f3r import geometry as GEO
from positioning.b2f3r import noise as NZ
from positioning.b2f3r.estimator import B2f3R
from positioning.evaluation.benchmark import with_estimator_profile
from positioning.models.v3_candidate import V3Candidate
from positioning.system.simulator import subsample_beams


def _sample(i, G, **kw):
    s, _ = D.draw("unit_test", i, **kw)
    return with_estimator_profile(subsample_beams(s, G), "strong_classical")


def test_namespace_is_new():
    from positioning.robust import data as RD
    assert D.NAMESPACE != RD.NAMESPACE and D.seeds("a", 0) != RD.seeds("a", 0)
    assert all(v >> 63 == 1 for v in D.seeds("a", 0).values())


@pytest.mark.parametrize("i,G", [(0, 8), (1, 16), (2, 16), (5, 32), (8, 6)])
def test_default_configuration_reproduces_the_frozen_estimator(i, G):
    s = _sample(i, G)
    a = V3Candidate(it_mid=3).estimate(s.y, s.context)
    b = B2f3R().estimate(s.y, s.context)
    assert np.array_equal(a.position, b.position) and a.orientation == b.orientation


def test_components_above_g_low_and_outside_olos_do_not_act():
    s = _sample(0, 16)                                   # index 0: LOS
    a = V3Candidate(it_mid=3).estimate(s.y, s.context)
    b = B2f3R(lowg="A2b", geo="S1").estimate(s.y, s.context)
    assert np.array_equal(a.position, b.position)


@pytest.mark.parametrize("noise", ["E2", "resid"])
def test_noise_components_do_not_read_the_true_noise_variance(noise):
    s = _sample(2, 16)
    est = B2f3R(noise=noise, lowg="A2b", geo="S1")
    a = est.estimate(s.y, s.context)
    b = est.estimate(s.y, dataclasses.replace(s.context, n0=s.context.n0 * 37.0))
    assert np.array_equal(a.position, b.position) and a.info["n0_used"] == b.info["n0_used"]
    assert 0.5 < a.info["n0_used"] / s.context.n0 < 2.0


def test_residual_floor_is_unbiased_on_noise_and_scale_free():
    s = _sample(0, 16)
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(s.y.shape) + 1j * rng.standard_normal(s.y.shape)) * np.sqrt(0.5 * 3.7)
    assert abs(NZ.residual_floor(w, s.context) / 3.7 - 1) < 0.03
    assert np.isclose(NZ.residual_floor(10 * w, s.context), 100 * NZ.residual_floor(w, s.context))
    M = w.size
    assert np.sum(np.abs(w) ** 2) / NZ.residual_floor(w, s.context) < NZ.ratio_threshold(M) * 1.02


def test_geometry_rule_keeps_the_learned_fit_on_ties():
    s = _sample(2, 32)                                   # OLOS
    c = V3Candidate(it_mid=3)
    from positioning.b2f3r import lowg as LW
    est, f, log, cap = LW.path_stage(c, s.y, s.context, "frozen")
    a = GEO.localize(est, s.y, s.context, c.geom, "frozen")
    b = GEO.localize(est, s.y, s.context, c.geom, "S1")
    if b.info["geo"] in ("learned", "frozen"):
        assert np.allclose(a.position, b.position)
    else:
        assert b.cost < a.cost - GEO.C0

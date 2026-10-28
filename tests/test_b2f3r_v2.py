"""B2f3R-v2: default = frozen B2f3; v1 configuration = B2f3R-v1; no use of true N0 or of the condition label."""
import dataclasses

import numpy as np
import pytest

from positioning.b2f3r.estimator import B2f3R
from positioning.b2f3r_v2 import data as D
from positioning.b2f3r_v2 import noise_support as NS
from positioning.b2f3r_v2.candidate import B2f3Rv2
from positioning.evaluation.benchmark import with_estimator_profile
from positioning.models.v3_candidate import V3Candidate
from positioning.system.simulator import subsample_beams


def _sample(i, G, region="iid"):
    s, _ = D.draw("unit_test_v2", i, region=region)
    return with_estimator_profile(subsample_beams(s, G), "strong_classical")


def test_namespace_is_new():
    from positioning.b2f3r import data as D1
    assert D.NAMESPACE not in (D1.NAMESPACE,) and D.seeds("a", 0) != D1.seeds("a", 0)


@pytest.mark.parametrize("i,G", [(0, 8), (2, 16), (5, 32), (8, 6)])
def test_all_off_is_frozen(i, G):
    s = _sample(i, G)
    a = V3Candidate(it_mid=3).estimate(s.y, s.context)
    b = B2f3Rv2().estimate(s.y, s.context)
    assert np.array_equal(a.position, b.position) and a.orientation == b.orientation


@pytest.mark.parametrize("i,G", [(1, 8), (2, 16), (5, 6), (11, 32)])
def test_v1_configuration_is_v1(i, G):
    s = _sample(i, G, "lat")
    a = B2f3R(noise="resid", lowg="A2b", geo="S1").estimate(s.y, s.context)
    b = B2f3Rv2(noise="resid", lowg="A2b", geo="S1").estimate(s.y, s.context)
    assert np.array_equal(a.position, b.position) and a.orientation == b.orientation


@pytest.mark.parametrize("noise", ["censored", "cfar"])
def test_no_true_n0_and_no_condition_label(noise):
    s = _sample(2, 16)
    est = B2f3Rv2(noise=noise, lowg="A2b", geo="S1")
    a = est.estimate(s.y, s.context)
    other = "los" if s.context.config.scenario.condition != "los" else "olos"
    ctx = dataclasses.replace(s.context, n0=s.context.n0 * 41.0, config=s.context.config.replace(**{"scenario.condition": other}))
    b = est.estimate(s.y, ctx)
    assert np.array_equal(a.position, b.position) and a.info["n0_used"] == b.info["n0_used"]


def test_censored_floor_unbiased_and_scale_equivariant():
    s = _sample(0, 16)
    rng = np.random.default_rng(1)
    w = (rng.standard_normal(s.y.shape) + 1j * rng.standard_normal(s.y.shape)) * np.sqrt(0.5 * 2.3)
    k = NS.constants(16)["kappa"]
    assert abs(NS.censored_floor(w, s.context, k) / 2.3 - 1) < 0.03
    assert np.isclose(NS.censored_floor(5 * w, s.context, k), 25 * NS.censored_floor(w, s.context, k))
    assert NS.stop_ok("cfar", float(np.sum(np.abs(w) ** 2)), NS.censored_floor(w, s.context, k), w, s.context) in (True, False)


def test_calibration_covers_all_beam_counts():
    for G in (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32):
        c = NS.constants(G)
        assert 0.98 < c["kappa"] < 1.02 and 2.0 < c["a99"] < 3.0 and c["draws"] >= 6000

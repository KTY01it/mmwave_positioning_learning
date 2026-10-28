"""B2f3R-v3: all components off = B2f3R-v2 bit for bit; exact runtime controls; no use of true N0 or of the condition
label; the ambiguity flag never changes the estimate."""
import dataclasses

import numpy as np
import pytest

from positioning.b2f3r_v2 import data as D2
from positioning.b2f3r_v2.candidate import B2f3Rv2
from positioning.b2f3r_v3 import data as D
from positioning.b2f3r_v3.candidate import B2f3Rv3
from positioning.evaluation.benchmark import with_estimator_profile
from positioning.system.simulator import subsample_beams

R1 = dict(hyps=("Hp", "Hm"), select="exip_pen")


def _sample(i, G, region="iid", condition=None):
    s = None
    while s is None:                                   # region draws can be rejected: next index
        s, _ = D.draw("unit_test_v3", i, region=region, condition=condition, snr=("uniform", -10.0, 10.0))
        i += 1000
    return with_estimator_profile(subsample_beams(s, G), "strong_classical")


def _same(a, b):
    return np.array_equal(a.position, b.position) and a.orientation == b.orientation


def test_namespace_is_new():
    assert D.NAMESPACE != D2.NAMESPACE and D.seeds("a", 0) != D2.seeds("a", 0)


@pytest.mark.parametrize("i,G,region,cond", [(0, 8, "iid", None), (2, 16, "iid", None), (3, 16, "latneg", "olos"),
                                              (4, 32, "latneg", "olos"), (5, 6, "lat", None)])
def test_all_off_is_v2(i, G, region, cond):
    s = _sample(i, G, region, cond)
    a = B2f3Rv2(noise="resid", lowg="A2b", geo="S1", support="G2").estimate(s.y, s.context)
    b = B2f3Rv3().estimate(s.y, s.context)
    assert _same(a, b) and b.info["v3"]["triggered"] == 0


@pytest.mark.parametrize("i,G", [(3, 16), (4, 32), (6, 16), (7, 8)])
def test_d4_is_exact(i, G):
    s = _sample(i, G, "latneg", "olos")
    a = B2f3Rv3(**R1, d4=False).estimate(s.y, s.context)
    b = B2f3Rv3(**R1, d4=True).estimate(s.y, s.context)
    assert _same(a, b) and a.info["v3"]["selected"] == b.info["v3"]["selected"]


def test_no_true_n0_and_no_condition_label():
    hits = 0
    for i in range(3, 9):
        s = _sample(i, 16, "latneg", "olos")
        est = B2f3Rv3(**R1)
        a = est.estimate(s.y, s.context)
        other = "los" if s.context.config.scenario.condition != "los" else "nlos"
        ctx = dataclasses.replace(s.context, n0=s.context.n0 * 41.0,
                                  config=s.context.config.replace(**{"scenario.condition": other}))
        b = est.estimate(s.y, ctx)
        assert _same(a, b) and a.info["n0_used"] == b.info["n0_used"] and all(a.info["v3"].get(k) == b.info["v3"].get(k) for k in ("selected", "scores", "skipped", "ambiguous"))
        hits += a.info["v3"]["triggered"]
    assert hits > 0


def test_flag_never_replaces_estimate():
    s = _sample(4, 32, "latneg", "olos")
    a = B2f3Rv3(**R1, flag_theta=0.0).estimate(s.y, s.context)
    b = B2f3Rv3(**R1, flag_theta=np.inf).estimate(s.y, s.context)
    assert _same(a, b) and np.all(np.isfinite(a.position))


def test_component_switches_are_independent():
    s = _sample(3, 16, "latneg", "olos")
    assert _same(B2f3Rv3(hyps=()).estimate(s.y, s.context), B2f3Rv3(hyps=(), multistart=True).estimate(s.y, s.context))

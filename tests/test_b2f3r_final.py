"""Finalization: all new options off = frozen v3.1; the scenario block of the configuration (UE box, orientation prior,
scatterer prior, condition) and the simulator N0 do not influence any candidate; budget/model-order options are inference-only."""
import dataclasses

import numpy as np
import pytest

from positioning.b2f3r_final import data as D
from positioning.b2f3r_final.candidate import B2f3RFinal
from positioning.b2f3r_v31 import data as D31
from positioning.b2f3r_v31.candidate import B2f3Rv31
from positioning.evaluation.benchmark import with_estimator_profile
from positioning.system.simulator import subsample_beams

V31 = B2f3Rv31(gate="deterministic", recovery="combined", runtime="cached")


def _sample(i, G, region="latneg", condition="olos", snr=("uniform", -10.0, 10.0)):
    s = None
    while s is None:
        s, _ = D.draw("unit_test_final", i, region=region, condition=condition, snr=snr)
        i += 1000
    return with_estimator_profile(subsample_beams(s, G), "strong_classical")


def _same(a, b):
    return np.array_equal(a.position, b.position) and a.orientation == b.orientation


def test_namespace_is_new():
    assert D.NAMESPACE != D31.NAMESPACE and D.seeds("a", 0) != D31.seeds("a", 0)


@pytest.mark.parametrize("i,G,region,cond", [(0, 16, "latneg", "olos"), (1, 32, "latneg", "olos"), (2, 8, "iid", None), (3, 16, "lat", "olos")])
def test_all_off_is_v31(i, G, region, cond):
    st = _sample(i, G, region, cond)
    assert _same(V31.estimate(st.y, st.context), B2f3RFinal().estimate(st.y, st.context))


@pytest.mark.parametrize("est", [V31, B2f3RFinal(relprune_rho=15.0), B2f3RFinal(max_loc=1)])
def test_scenario_block_and_n0_not_used(est):
    """Replace the entire scenario configuration (receiver box, orientation prior, scatterer prior, condition) by that of a
    different region and scale N0 by 41: identical output."""
    st = _sample(5, 16, "latneg", "olos", ("choice", [20.0]))
    other = _sample(6, 16, "far", "los")
    cfg2 = dataclasses.replace(st.context.config, scenario=other.context.config.scenario)
    ctx2 = dataclasses.replace(st.context, n0=st.context.n0 * 41.0, config=cfg2)
    assert st.context.config.scenario != cfg2.scenario
    a, b = est.estimate(st.y, st.context), est.estimate(st.y, ctx2)
    assert _same(a, b)


def test_budget_unbounded_equals_v31():
    st = _sample(7, 16)
    assert _same(V31.estimate(st.y, st.context), B2f3RFinal(max_loc=10).estimate(st.y, st.context))


from positioning.b2f3r_final.frames import BSFrame  # noqa: E402
from positioning.b2f3r_v3.candidate import B2f3Rv3  # noqa: E402


@pytest.mark.parametrize("est", [BSFrame(V31), BSFrame(B2f3RFinal(relprune_rho=15.0)), BSFrame(B2f3Rv3())])
def test_bs_frame_translation_equivariance(est):
    t = np.array([1.3, -0.7])
    for i in range(3):
        st = _sample(20 + i, 16)
        a = est.estimate(st.y, st.context)
        b = est.estimate(st.y, dataclasses.replace(st.context, bs_position=np.asarray(st.context.bs_position, float) + t))
        assert np.allclose(b.position - t, a.position, atol=1e-6) and abs(np.angle(np.exp(1j * (b.orientation - a.orientation)))) < 1e-6


def test_bs_frame_is_identity_at_origin():
    st = _sample(30, 16)
    assert _same(BSFrame(V31).estimate(st.y, st.context), V31.estimate(st.y, st.context))

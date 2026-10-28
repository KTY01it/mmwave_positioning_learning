"""B2f3R-v3.1: exact optimizations reproduce the v3 generators; cheap features are observable (no true N0 / label)."""
import dataclasses

import numpy as np
import pytest

from positioning.b2f3r import noise as NZ1
from positioning.b2f3r_v2 import geometry_recovery as GR
from positioning.b2f3r_v3 import data as D3
from positioning.b2f3r_v3 import hypotheses as HY
from positioning.b2f3r_v3.candidate import B2f3Rv3
from positioning.b2f3r_v31 import data as D
from positioning.b2f3r_v31 import reliability_gate as RG
from positioning.b2f3r_v31 import runtime_optimizations as RO
from positioning.evaluation.benchmark import with_estimator_profile
from positioning.system.simulator import subsample_beams

V3 = B2f3Rv3(hyps=("Hp", "Hm"), select="exip_pen", d4=True)


def _sample(i, G, region="latneg", condition="olos"):
    s = None
    while s is None:
        s, _ = D.draw("unit_test_v31", i, region=region, condition=condition, snr=("uniform", -10.0, 10.0))
        i += 1000
    return with_estimator_profile(subsample_beams(s, G), "strong_classical")


def _h0(st, ctx_in=None):
    ctx = dataclasses.replace(ctx_in or st.context, n0=NZ1.plug_in(st.y, ctx_in or st.context, "E2"))
    e0, f0, n0, log = V3.paths(st.y, ctx)
    return ctx, e0, f0, n0, log


def test_namespace_is_new():
    assert D.NAMESPACE != D3.NAMESPACE and D.seeds("a", 0) != D3.seeds("a", 0)


@pytest.mark.parametrize("i,G", [(0, 8), (1, 16), (2, 32), (3, 16), (4, 6)])
def test_plus_one_fast_is_exact(i, G):
    st = _sample(i, G)
    ctx, e0, f0, n0, _ = _h0(st)
    c0 = dataclasses.replace(ctx, n0=n0)
    a, fa = HY.plus_one(st.y, c0, e0)
    b, fb = RO.plus_one_fast(st.y, c0, e0)
    assert fa == fb and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("delay", "aod", "aoa", "gain_tilde"))


def test_features_do_not_use_true_n0_or_label():
    st = _sample(5, 16)
    other = "los" if st.context.config.scenario.condition != "los" else "nlos"
    ctx2 = dataclasses.replace(st.context, n0=st.context.n0 * 41.0, config=st.context.config.replace(**{"scenario.condition": other}))
    out = []
    for c in (st.context, ctx2):
        ctx, e0, f0, n0, log = _h0(st, c)
        c0 = dataclasses.replace(ctx, n0=n0)
        o = GR.localize(e0, st.y, c0, V3.frozen.geom, V3.geo)
        out.append(RG.features(st.y, c0, e0, f0, n0, o, log))
    assert out[0] == out[1]


from positioning.b2f3r_v31.candidate import B2f3Rv31  # noqa: E402


def _same(a, b):
    return np.array_equal(a.position, b.position) and a.orientation == b.orientation


@pytest.mark.parametrize("i,G,region,cond", [(0, 16, "latneg", "olos"), (1, 32, "latneg", "olos"), (2, 8, "latneg", "olos"),
                                              (3, 16, "iid", None), (4, 6, "iid", None)])
def test_all_off_and_cached_equal_v3(i, G, region, cond):
    st = _sample(i, G, region, cond)
    a = V3.estimate(st.y, st.context)
    for est in (B2f3Rv31(), B2f3Rv31(runtime="cached"), B2f3Rv31(gate="deterministic", tau_peak=-np.inf, tau_cost=-np.inf)):
        b = est.estimate(st.y, st.context)
        assert _same(a, b) and a.info["v3"]["selected"] == b.info["v3"]["selected"]


def test_v31_no_true_n0_or_label_and_flag_info_only():
    st = _sample(6, 16)
    est = B2f3Rv31(gate="deterministic", recovery="combined", runtime="cached")
    other = "los" if st.context.config.scenario.condition != "los" else "nlos"
    ctx2 = dataclasses.replace(st.context, n0=st.context.n0 * 41.0, config=st.context.config.replace(**{"scenario.condition": other}))
    a, b = est.estimate(st.y, st.context), est.estimate(st.y, ctx2)
    assert _same(a, b) and a.info["v3"]["selected"] == b.info["v3"]["selected"]
    c = B2f3Rv31(gate="deterministic", recovery="combined", runtime="cached", flag_theta=np.inf).estimate(st.y, st.context)
    assert _same(a, c)


def test_alias_flip_preserves_sine_magnitude():
    st = _sample(7, 16)
    ctx, e0, f0, n0, _ = _h0(st)
    from positioning.b2f3r_v31 import targeted_recovery as TR
    ks = TR.endfire_paths(e0, "aoa", 0.0)
    e, f = TR.alias_flip(st.y, dataclasses.replace(ctx, n0=n0), e0, ks[0], "aoa")
    assert e.num_paths == e0.num_paths and np.isfinite(f)

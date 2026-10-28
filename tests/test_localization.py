"""Position/orientation estimation: LOS (61)-(62), NLOS EXIP (63), OLOS (64)-(65), unknown condition."""
import numpy as np
import pytest

from positioning.config import load_config
from positioning.localization.nlos import nlos_localize
from positioning.localization.olos import olos_localize
from positioning.localization.pipeline import ClassicalEstimator, OracleInitEstimator
from positioning.models.base import Estimator
from positioning.signal_processing.model import ChannelEstimate
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


def _sample(name, snr=200.0, seed=0, **over):
    cfg = load_config(CONFIGS / "experiments" / name, **over)
    return MmWavePositioningSystem(cfg).simulate(seed=seed, snr_db=snr)


def test_nlos_exip_exact_parameters():
    s = _sample("nlos.yaml")
    r = nlos_localize(ChannelEstimate.from_paths(s.gt.paths), s.context, "fim")
    np.testing.assert_allclose(r.position, s.gt.position, atol=1e-9)
    assert r.orientation == pytest.approx(s.gt.orientation, abs=1e-9)
    np.testing.assert_allclose(r.scatterers, s.gt.scatterers, atol=1e-8)


def test_olos_exip_exact_parameters():
    s = _sample("olos.yaml")
    r = olos_localize(ChannelEstimate.from_paths(s.gt.paths), s.context, "fim", alpha_step=0.05, top_k=5)
    np.testing.assert_allclose(r.position, s.gt.position, atol=1e-6)
    assert r.orientation == pytest.approx(s.gt.orientation, abs=1e-6)


@pytest.mark.parametrize("name", ["los.yaml", "nlos.yaml"])
def test_pipeline_high_snr(name):
    s = _sample(name, snr=20.0, seed=3)
    out = ClassicalEstimator()(s)
    assert np.linalg.norm(out.position - s.gt.position) < 2e-3
    assert abs(out.orientation - s.gt.orientation) < 1e-3
    assert not out.oracle


def test_oracle_estimator_is_tagged_and_gets_gt():
    s = _sample("los.yaml", snr=0.0)
    out = OracleInitEstimator()(s)
    assert out.oracle and np.linalg.norm(out.position - s.gt.position) < 0.05


def test_non_oracle_estimator_never_sees_ground_truth():
    seen = {}

    class Spy(Estimator):
        def estimate(self, y, context, gt=None):
            seen["gt"] = gt
            from positioning.models.base import PositionEstimate
            return PositionEstimate(np.zeros(2), 0.0)
    Spy()(_sample("los.yaml"))
    assert seen["gt"] is None


def test_unknown_condition_picks_nlos_for_nlos_data():
    s = _sample("nlos.yaml", snr=10.0, **{"scenario.scatterers": {"mode": "fixed", "positions": [[1.5, 0.4], [1.5, 0.9]]}})
    out = ClassicalEstimator("unknown")(s)
    assert out.condition == "nlos"


def test_nlos_pruning_removes_inconsistent_path():
    """F24: a strong path that no single-bounce scatterer explains is pruned; the true geometry is recovered."""
    s = _sample("nlos.yaml", snr=10.0)
    tr = ChannelEstimate.from_paths(s.gt.paths)
    fake = ChannelEstimate(np.r_[tr.delay, 54e-9], np.r_[tr.aod, 0.39], np.r_[tr.aoa, -2.67],
                           np.r_[tr.gain_tilde, 2 * tr.gain_tilde[1]])
    raw = nlos_localize(fake, s.context, "fim", "strongest", prune_level=None)
    pruned = nlos_localize(fake, s.context, "fim", "strongest", prune_level=1e-3)
    assert np.linalg.norm(raw.position - s.gt.position) > 1e-3
    np.testing.assert_allclose(pruned.position, s.gt.position, atol=1e-6)
    assert sorted(pruned.kept_paths) == [0, 1]


def test_unknown_condition_with_pruning_still_detects_olos():
    """Regression: chi2 pruning must not be applied inside the NLOS-vs-OLOS hypothesis test."""
    s = _sample("olos.yaml", snr=0.0, seed=2, **{"estimator.olos_alpha_step": 0.1})
    out = ClassicalEstimator("unknown")(s)
    assert out.condition == "olos"


@pytest.mark.parametrize("kw", [{}, {"oracle_geometry": True}, {"observable_only": False}])
def test_oracle_variants_run_on_nlos_and_olos(kw):
    for name in ("nlos.yaml", "olos.yaml"):
        s = _sample(name, snr=10.0, **{"estimator.olos_alpha_step": 0.1})
        out = OracleInitEstimator(**kw)(s)
        assert out.oracle and set(out.info["timing"]) == {"coarse", "sage", "localize"}
        assert np.linalg.norm(out.position - s.gt.position) < 1.0

"""P2 harness: paired G subsets, bounds per G, failures kept, oracle flagged."""
import numpy as np

from positioning.config import load_config
from positioning.evaluation.benchmark import evaluate_sample, summarize, with_estimator_profile
from positioning.localization.pipeline import ClassicalEstimator, OracleInitEstimator
from positioning.models.base import Estimator, PositionEstimate
from positioning.system.simulator import MmWavePositioningSystem
from conftest import CONFIGS


def _sample(snr=0.0, seed=0):
    return MmWavePositioningSystem(load_config(CONFIGS / "experiments" / "nlos.yaml")).simulate(seed=seed, snr_db=snr)


def test_records_per_G_and_bound_grows_with_fewer_beams():
    recs = evaluate_sample(ClassicalEstimator(), _sample(), G_list=[8, 32], method="strong")
    assert [r["G"] for r in recs] == [8, 32]
    assert recs[0]["peb"] > recs[1]["peb"]  # fewer beams -> less information
    assert all(np.isfinite(r["eps_pos"]) and not r["oracle"] for r in recs)
    rows = summarize(recs)
    assert {r["G"] for r in rows} == {8, 32} and all(r["n"] == 1 for r in rows)


def test_failures_are_recorded_not_dropped():
    class Broken(Estimator):
        def estimate(self, y, context, gt=None):
            raise RuntimeError("boom")
    rec = evaluate_sample(Broken(), _sample(), method="broken")[0]
    assert rec["failed"] and "boom" in rec["error"]
    assert summarize([rec])[0]["p_out"] == 1.0


def test_oracle_flag_and_profile_switch():
    s = _sample()
    assert evaluate_sample(OracleInitEstimator(), s)[0]["oracle"]
    se = with_estimator_profile(s, "shahmansoori_exact")
    assert se.context.config.estimator.detection == "dcs_somp" and np.array_equal(se.y, s.y)


def test_paper_exact_estimator_runs_through_the_harness():
    rec = evaluate_sample(ClassicalEstimator(), with_estimator_profile(_sample(snr=-25.0), "shahmansoori_exact"))[0]
    assert np.isfinite(rec["pos_err"])

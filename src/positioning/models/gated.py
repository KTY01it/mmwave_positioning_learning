"""Gated proposed estimator (step 6C, frozen): fast learned pipeline; if the consistency gate rejects it, the fixed
fallback strong classical v2.2 runs on the same observation. Thresholds are read from the locked gate file."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT
import dataclasses

import yaml

from ..config import CONFIG_DIR
from ..localization.pipeline import ClassicalEstimator
from .base import Estimator, PositionEstimate
from .proposed import ProposedEstimator


def _strong_context(context):
    """Same observation context with the frozen fallback estimator settings (strong classical v2.2)."""
    prof = yaml.safe_load((CONFIG_DIR / "estimators" / "strong_classical.yaml").read_text())["estimator"]
    cfg = context.config.replace(**{f"estimator.{k}": v for k, v in prof.items() if k != "olos_alpha_step"})
    return dataclasses.replace(context, config=cfg)


def load_locked_gate(lock_path: Path = REPO_ROOT / "docs" / "step6c_gate_lock.json"):
    lock = json.loads(Path(lock_path).read_text())
    f = REPO_ROOT / lock["gate_file"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != lock["sha256"]:
        raise RuntimeError("gate.json changed after locking")
    g = json.loads(f.read_text())
    return g["chosen"], {int(k): v for k, v in g["thresholds"].items()}


def thresholds_for(G: int, table: dict) -> dict:
    """Calibrated G -> its thresholds; unseen G -> those of the largest calibrated G <= G (conservative: fewer beams
    were calibrated more strictly); below the smallest calibrated G -> the smallest."""
    keys = sorted(table)
    below = [k for k in keys if k <= G]
    return table[below[-1] if below else keys[0]]


class GatedProposedEstimator(Estimator):
    requires_oracle = False
    name = "gated_proposed"

    def __init__(self, path_ckpt: str, geom_ckpt: str, condition: str = "auto"):
        self.fast = ProposedEstimator(path_ckpt, geom_ckpt, sage_iterations=1, condition=condition)
        self.variant, self.table = load_locked_gate()
        self.condition = condition

    def accept(self, gate: dict, G: int) -> bool:
        th = thresholds_for(G, self.table)
        z_ok, v_ok = gate["z"] <= th["z"], gate["v"] <= th["v"]
        return {"z": z_ok, "v": v_ok, "zv": z_ok and v_ok}[self.variant]

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        out = self.fast.estimate(y, context)
        t_fast = time.perf_counter() - t0
        if self.accept(out.info["gate"], y.shape[1]):
            out.info["gated"] = {"accepted": True, "t_fast": t_fast, "t_fallback": 0.0}
            return out
        t1 = time.perf_counter()
        fb = ClassicalEstimator(self.condition).estimate(y, _strong_context(context))
        fb.info["gated"] = {"accepted": False, "t_fast": t_fast, "t_fallback": time.perf_counter() - t1,
                            "fast_gate": out.info["gate"]}
        fb.info.setdefault("timing", {})["gate_fast_path"] = t_fast
        return fb

"""B2f3R-v2 candidate estimator. Frozen B2f3 networks, thresholds and localisation branches; every intervention can be
switched off (all off = V3Candidate(it_mid=3); noise="resid", lowg="A2b", geo="S1", support=None = B2f3R-v1).

  noise    "known" (context.n0 as given: the frozen assumption) | "resid" (N1, v1) | "censored" (N2) | "cfar" (N3)
  lowg     None | "A2b"   selective classical coarse support for G <= g_low (T1, v1 component reused)
  geo      "frozen" | "S1" observation-based geometric-start choice (G1, v1 component reused)
  support  None | "G2"    selective multipath-support repair for G > g_low (T2, new) [DESIGN]

Inference inputs: y, pilots x, beams F, array/system configuration, BS position. The context's n0 is replaced by an
observation-based value unless noise="known"; the condition stored in context.config.scenario is never read
(tests/test_b2f3r_v2.py swaps it)."""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..b2f3r import lowg as LW1
from ..b2f3r import noise as NZ1
from ..models.base import Estimator, PositionEstimate
from ..models.path_init import network_input
from ..models.v3_candidate import V3Candidate
from ..robust import lowg as LG
from . import geometry_recovery as GR
from .path_recovery import model_order_loop

LOBE_T = 1.0
G_LOW = 10


class B2f3Rv2(Estimator):
    requires_oracle = False

    def __init__(self, noise: str = "known", lowg: str | None = None, geo: str = "frozen", support: str | None = None,
                 g_low: int = G_LOW, it_mid: int = 3):
        self.frozen = V3Candidate(it_mid=it_mid)
        self.noise, self.lowg, self.geo, self.support, self.g_low = noise, lowg, geo, support, g_low
        self.name = f"b2f3r_v2[noise={noise},lowg={lowg},geo={geo},support={support}]"

    def paths(self, y, context, force_coarse: bool = False, tm: dict | None = None):
        tm = {} if tm is None else tm
        low = self.lowg if y.shape[1] <= self.g_low else None
        log = {}
        t0 = time.perf_counter()
        x = network_input(y, context)
        kw = dict(it_mid=self.frozen.it_mid, noise=self.noise, log=log)
        if force_coarse:
            est, f, n0 = model_order_loop(self.frozen.pe, y, context, x, cand_fn=LG.cf_hybrid, **kw)
            tm["loop"] = tm.get("loop", 0.0) + time.perf_counter() - t0
            return est, f, n0, log
        est, f, n0 = model_order_loop(self.frozen.pe, y, context, x, switch=low == "A2b", **kw)
        tm["loop"] = tm.get("loop", 0.0) + time.perf_counter() - t0
        log["lobe_rerun"] = 0
        if low == "A2b" and not log["switched"]:
            t1 = time.perf_counter()
            lr = max(LW1.lobe_ratios(est, y, context))
            tm["lobe"] = time.perf_counter() - t1
            if lr > LOBE_T:
                t1 = time.perf_counter()
                first = dict(log)
                est, f, n0 = model_order_loop(self.frozen.pe, y, dataclasses.replace(context, n0=n0), x,
                                              cand_fn=LG.cf_hybrid, **kw)
                for k in ("n_sage", "sweeps", "rounds", "fwd_entered", "accepted", "removed", "fwd_tests", "level_evals"):
                    log[k] = log[k] + first[k]
                log["lobe_rerun"] = 1
                tm["rerun"] = time.perf_counter() - t1
        return est, f, n0, log

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        tm = {}
        if self.noise != "known":
            context = dataclasses.replace(context, n0=NZ1.plug_in(y, context, "E2"))
        tm["input"] = time.perf_counter() - t0
        est, f, n0, log = self.paths(y, context, tm=tm)
        t1 = time.perf_counter()
        ctx = dataclasses.replace(context, n0=n0)
        out = GR.localize(est, y, ctx, self.frozen.geom, self.geo)
        tm["localize"] = time.perf_counter() - t1
        log["support_repair"] = 0
        if self.support == "G2" and y.shape[1] > self.g_low:
            t2 = time.perf_counter()
            out, rep = GR.support_repair(self, out, est, f, n0, y, context, log)
            log["support_repair"] = rep
            tm["support"] = time.perf_counter() - t2
        tm["total"] = time.perf_counter() - t0
        out.info.update({"loop": log, "n0_used": float(n0), "n0_in": float(context.n0), "timing_v2": tm})
        return out

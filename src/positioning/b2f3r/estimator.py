"""B2f3-Robust candidate estimator: the frozen B2f3 networks, thresholds and localisation branches, with optional
observation-only components. Every component can be switched off; with all of them off the output equals V3Candidate.

  noise  None      context.n0 as given (the simulator's exact N0: the frozen assumption)
         "E2"/"E3" plug-in noise-floor estimate from the observation, used everywhere
         "resid"   E2 for the network input, then the residual-floor procedure of b2f3r.noise in the loop; the final
                   level is used by the condition test and the geometry stage
  lowg   None | "A1" (classical coarse support always) | "A2a" (selective, stop criterion) | "A2b" (A2a + lobe check),
         applied only when G <= g_low
  geo    "frozen" | "S1" (cost-margin multi-start of the OLOS branch, b2f3r.geometry)
"""
from __future__ import annotations

import dataclasses
import time

from ..models.base import Estimator, PositionEstimate
from ..models.path_init import network_input
from ..models.v3_candidate import V3Candidate
from ..robust import lowg as LG
from . import geometry as GEO
from . import loop as LOOP
from . import lowg as LW
from . import noise as NZ

LOBE_T = 1.0      # A2b: a transmit lobe outside the path's own cell explains the residual better


class B2f3R(Estimator):
    requires_oracle = False

    def __init__(self, noise: str | None = None, lowg: str | None = None, geo: str = "frozen", g_low: int = LW.G_LOW,
                 it_mid: int = 3):
        self.frozen = V3Candidate(it_mid=it_mid)
        self.noise, self.lowg, self.geo, self.g_low = noise, lowg, geo, g_low
        self.name = "b2f3r[" + ",".join(f"{k}={v}" for k, v in (("noise", noise), ("lowg", lowg), ("geo", geo))) + "]"

    def paths(self, y, context):
        """(est, n0, log) of the path stage. context.n0 must already be the noise input."""
        low = self.lowg if y.shape[1] <= self.g_low else None
        mode = "resid" if self.noise == "resid" else "known"
        log = {}
        x = network_input(y, context)
        kw = dict(it_mid=self.frozen.it_mid, noise=mode, log=log)
        est, f, n0 = LOOP.model_order_loop(self.frozen.pe, y, context, x, switch=low in ("A2a", "A2b"),
                                           cand_fn=LG.cf_hybrid if low == "A1" else None, **kw)
        log["lobe_rerun"] = 0
        if low == "A2b" and not log["switched"] and max(LW.lobe_ratios(est, y, context)) > LOBE_T:
            first = dict(log)
            est, f, n0 = LOOP.model_order_loop(self.frozen.pe, y, dataclasses.replace(context, n0=n0), x,
                                               cand_fn=LG.cf_hybrid, **kw)
            log.update({k: log[k] + first[k] for k in ("n_sage", "sweeps", "rounds", "fwd_entered", "accepted", "removed",
                                                       "fwd_tests")})
            log["lobe_rerun"] = 1
        return est, n0, log

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        if self.noise is not None:
            context = dataclasses.replace(context, n0=NZ.plug_in(y, context, "E2" if self.noise == "resid" else self.noise))
        est, n0, log = self.paths(y, context)
        t1 = time.perf_counter()
        ctx = dataclasses.replace(context, n0=n0)
        out = GEO.localize(est, y, ctx, self.frozen.geom, self.geo)
        log["fwd_entered"] = int(log["fwd_entered"])
        out.info.update({"loop": log, "n0_used": float(n0), "n0_in": float(context.n0),
                         "timing": {"paths": t1 - t0, "localize": time.perf_counter() - t1}})
        return out

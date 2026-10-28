"""Finalization candidates built on frozen B2f3R-v3.1. All new options off = B2f3Rv31(gate="deterministic",
recovery="combined", runtime="cached") bit for bit (tests/test_b2f3r_final.py).

  relprune_rho  None | float   F2-M1: one extra hypothesis removing every path more than rho dB weaker than the strongest
                               (b2f3r_final.model_order.relprune), compared with the frozen B4 score            [DESIGN]
  max_loc       None | int     F3: deterministic operation-count budget - at most `max_loc` extra hypotheses are localised,
                               in increasing order of their D4 bound (f/n0 + Z K); the others are skipped          [DESIGN]
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..b2f3r_v2 import geometry_recovery as GR
from ..b2f3r_v3 import hypotheses as HY
from ..b2f3r_v3.candidate import _cost, bound, score
from ..b2f3r_v31 import runtime_optimizations as RO  # noqa: F401
from ..b2f3r_v31 import targeted_recovery as TR
from ..b2f3r_v31.candidate import B2f3Rv31, plus_one_reuse, residual_peak
from . import model_order as MO

V31 = dict(gate="deterministic", recovery="combined", runtime="cached")


class B2f3RFinal(B2f3Rv31):
    def __init__(self, relprune_rho: float | None = None, max_loc: int | None = None, **kw):
        for k, v in V31.items():
            kw.setdefault(k, v)
        super().__init__(**kw)
        self.relprune_rho, self.max_loc = relprune_rho, max_loc
        self.name = f"b2f3r_final[v31,relprune={relprune_rho},max_loc={max_loc}]"

    def _hypotheses(self, out, base, h0, y, context):
        if self.relprune_rho is None and self.max_loc is None:
            return super()._hypotheses(out, base, h0, y, context)
        assert self.gate == "deterministic" and self.recovery == "combined" and self.runtime == "cached" and not self.prune
        e0, f0, n00 = h0
        eb, fb, _ = base
        ctx0 = dataclasses.replace(context, n0=n00)
        kind = self.select
        s_base = score(kind, fb, eb.num_paths, n00, _cost(out))
        cands = [("base", s_base, out)]
        info = {"triggered": 1, "scores": {"base": float(s_base)}, "skipped": {}, "t": {}, "generated": [], "gate": {}}
        t0 = time.perf_counter()
        pk = residual_peak(y, ctx0, e0)
        info["t"]["gate"] = time.perf_counter() - t0
        gens = []
        if pk[0] > self.tau_peak:
            gens.append(("Hp", lambda: plus_one_reuse(y, ctx0, e0, pk[2], pk[3], pk[1])))
        for k in TR.endfire_paths(eb, "aoa", self.endfire)[: self.max_flips]:
            gens.append((f"flip_aoa{k}", lambda k=k: TR.alias_flip(y, ctx0, eb, k, "aoa")))
        if _cost(out) > self.tau_cost:
            gens.append(("Hm", lambda: HY.minus_one(y, ctx0, e0)))
        if self.relprune_rho is not None:
            gens.append(("relprune", lambda: MO.relprune(y, ctx0, eb, self.relprune_rho)))
        made = []
        for h, fn in gens:
            t1 = time.perf_counter()
            res = fn()
            info["t"][h] = time.perf_counter() - t1
            if res is None:
                continue
            e, f = res
            info["generated"].append(h)
            if HY.same_support(e, e0, context):
                info["skipped"][h] = "duplicate"
            elif self.d4 and bound(kind, f, e.num_paths, n00) >= s_base:
                info["skipped"][h] = "bound"
            else:
                made.append((bound(kind, f, e.num_paths, n00), len(made), h, e, f))
        if self.max_loc is not None:
            made.sort(key=lambda q: (q[0], q[1]))
            for q in made[self.max_loc:]:
                info["skipped"][q[2]] = "budget"
            made = sorted(made[: self.max_loc], key=lambda q: q[1])
        for _, _, h, e, f in made:
            t1 = time.perf_counter()
            o = GR.localize(e, y, ctx0, self.frozen.geom, self.geo)
            info["t"][h] = info["t"].get(h, 0.0) + time.perf_counter() - t1
            if o.info.get("c_hat") != "olos":
                info["skipped"][h] = "not_olos"
                continue
            s = score(kind, f, e.num_paths, n00, _cost(o))
            info["scores"][h] = float(s)
            cands.append((h, s, o))
        order = sorted(range(len(cands)), key=lambda i: (cands[i][1], i))
        sel = cands[order[0]]
        margin = cands[order[1]][1] - sel[1] if len(order) > 1 else np.inf
        info.update({"selected": sel[0], "margin": float(margin), "ambiguous": int(sel[0] != "base" or margin < self.flag_theta),
                     "n_hyps": len(info["generated"]), "n_loc": len(cands) - 1})
        return sel[2], info

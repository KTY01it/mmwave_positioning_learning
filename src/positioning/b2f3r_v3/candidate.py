"""B2f3R-v3 candidate estimator. B2f3R-v2 (frozen B2f3 networks, thresholds and branches; residual-floor N0, A2b, S1,
G2) followed by an optional selective multipath-hypothesis stage. Every component can be switched off; all off =
B2f3Rv2(noise, lowg, geo, support) bit for bit (tests/test_b2f3r_v3.py).

  hyps       ()  | ("Hp", "Hm")   A: competing path supports for OLOS-routed outputs (hypotheses.plus_one / minus_one)
  select     "exip_pen" | "pen" | "geo"   B: score f/n0 + 0.5*EXIP cost + Z*K (B4, DESIGN), f/n0 + Z*K (B2), EXIP cost (B3)
  multistart False | True   C: minimum OLOS cost over S1 / learned / cls1 / cls3 for the selected support [rejected on v3val]
  d4         True | False   D: exact early exit -- an extra hypothesis whose bound f/n0 + Z*K already reaches the base score
                            cannot be selected, so it is not localised (output identical to d4=False)
  flag_theta 50.0           ambiguity flag (info only; the position estimate is never replaced)

Inference inputs: y, pilots x, beams F, array/system configuration, BS position. context.n0 is replaced by the E2 plug-in
(noise != "known"); the condition stored in context.config.scenario is never read."""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..b2f3r import geometry as GEO
from ..b2f3r import noise as NZ1
from ..b2f3r_v2 import geometry_recovery as GR
from ..b2f3r_v2.candidate import B2f3Rv2
from ..models.base import PositionEstimate
from . import hypotheses as HY

Z = GR.Z
FLAG_THETA = 50.0          # chosen on v3val (outputs/b2f3r_v3/flag_theta.json)
STARTS = ("learned", "cls1", "cls3")


def _cost(o) -> float:
    c = getattr(o, "cost", np.inf)
    return float(c) if c is not None and np.isfinite(c) else np.inf


def score(kind, f, k, n0, cost):
    if kind == "exip_pen":
        return f / n0 + 0.5 * cost + Z * k
    if kind == "pen":
        return f / n0 + Z * k
    if kind == "geo":
        return cost
    raise KeyError(kind)


def bound(kind, f, k, n0):
    """lower bound of `score` before localisation (EXIP cost >= 0); -inf when no bound exists."""
    return f / n0 + Z * k if kind in ("exip_pen", "pen") else -np.inf


class B2f3Rv3(B2f3Rv2):
    requires_oracle = False

    def __init__(self, noise: str = "resid", lowg: str | None = "A2b", geo: str = "S1", support: str | None = "G2",
                 hyps: tuple = (), select: str = "exip_pen", multistart: bool = False, d4: bool = True,
                 flag_theta: float = FLAG_THETA, g_min: int = 0, it_mid: int = 3):
        super().__init__(noise=noise, lowg=lowg, geo=geo, support=support, it_mid=it_mid)
        self.hyps, self.select, self.multistart, self.d4 = tuple(hyps), select, multistart, d4
        self.flag_theta, self.g_min = flag_theta, g_min
        self.name = (f"b2f3r_v3[noise={noise},lowg={lowg},geo={geo},support={support},hyps={'+'.join(self.hyps) or '-'},"
                     f"select={select},ms={int(multistart)},d4={int(d4)},g_min={g_min}]")

    # ---- v2 path (identical calls in identical order) -------------------------------------------------------------
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
        base = (est, f, n0)
        if self.support == "G2" and y.shape[1] > self.g_low:
            t2 = time.perf_counter()
            out, rep = self._repair(out, est, f, n0, y, context, log)
            if rep == 2:
                base = self._h1
            log["support_repair"] = rep
            tm["support"] = time.perf_counter() - t2
        v3 = {"triggered": 0, "selected": "base", "ambiguous": 0}
        if self.hyps and y.shape[1] >= self.g_min and out.info.get("c_hat") == "olos":
            t3 = time.perf_counter()
            out, v3 = self._hypotheses(out, base, (est, f, n0), y, context)
            tm["v3"] = time.perf_counter() - t3
        tm["total"] = time.perf_counter() - t0
        out.info.update({"loop": log, "n0_used": float(n0), "n0_in": float(context.n0), "timing_v2": tm, "v3": v3})
        return out

    def _repair(self, out, est, f, n0, y, context, log):
        """GR.support_repair with the repaired support kept for the hypothesis stage (same calls, same result)."""
        self._h1 = None
        if not GR.support_trigger(out):
            return out, 0
        est_h, f_h, n0_h, _ = self.paths(y, context, force_coarse=True)
        if not GR.choose(f, est.num_paths, f_h, est_h.num_paths, n0):
            return out, 1
        self._h1 = (est_h, f_h, n0_h)
        return GR.localize(est_h, y, dataclasses.replace(context, n0=n0_h), self.frozen.geom, self.geo), 2

    # ---- V3-A/B/C/D --------------------------------------------------------------------------------------------------
    def _hypotheses(self, out, base, h0, y, context):
        """A: Hp/Hm from the v2 path-stage support H0; B: selection against the support v2 used; D: exact skips."""
        e0, f0, n00 = h0
        eb, fb, _ = base
        ctx0 = dataclasses.replace(context, n0=n00)
        kind = self.select
        s_base = score(kind, fb, eb.num_paths, n00, _cost(out))
        cands = [("base", s_base, out)]
        ests = {"base": (eb, dataclasses.replace(context, n0=base[2]))}
        info = {"triggered": 1, "scores": {"base": float(s_base)}, "skipped": {}, "t": {}}
        gen = {"Hp": HY.plus_one, "Hm": HY.minus_one}
        for h in self.hyps:
            t0 = time.perf_counter()
            e, f = gen[h](y, ctx0, e0)
            if HY.same_support(e, e0, context):                                  # D3: duplicate of H0
                info["skipped"][h] = "duplicate"
            elif self.d4 and bound(kind, f, e.num_paths, n00) >= s_base:          # D4: cannot beat the base score
                info["skipped"][h] = "bound"
            else:
                o = GR.localize(e, y, ctx0, self.frozen.geom, self.geo)
                if o.info.get("c_hat") != "olos":
                    info["skipped"][h] = "not_olos"
                else:
                    s = score(kind, f, e.num_paths, n00, _cost(o))
                    info["scores"][h] = float(s)
                    cands.append((h, s, o))
                    ests[h] = (e, ctx0)
            info["t"][h] = time.perf_counter() - t0
        order = sorted(range(len(cands)), key=lambda i: (cands[i][1], i))
        sel = cands[order[0]]
        margin = cands[order[1]][1] - sel[1] if len(order) > 1 else np.inf
        info.update({"selected": sel[0], "margin": float(margin),
                     "ambiguous": int(sel[0] != "base" or margin < self.flag_theta)})
        res = sel[2]
        if self.multistart:
            res = self._multistart(res, *ests[sel[0]], y, info)
        return res, info

    def _multistart(self, o, e, ctx0, y, info):
        """C (rejected on v3val, kept switchable): the localisation of the selected support with the smallest OLOS cost
        among the S1 output and the raw learned / classical starts."""
        if e.num_paths < 2:
            return o
        es, fits = GEO.olos_fits(e, ctx0, self.frozen.geom, STARTS)
        best, bc = o, _cost(o)
        for v in STARTS:
            f, _ = fits[v]
            c = float(f.cost) if np.isfinite(f.cost) else np.inf
            if c < bc:
                best, bc = GEO.to_estimate(es, f, y, ctx0), c
                best.info.update({"c_hat": "olos", "geo": f"ms_{v}"})
        info["multistart"] = best.info.get("geo")
        return best

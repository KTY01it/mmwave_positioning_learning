"""B2f3R-v3.1 candidate estimator: B2f3R-v2 unchanged, then (for OLOS-routed outputs) a gated and extended hypothesis stage.
All new components off = B2f3R-v3 (B2f3Rv3(hyps=("Hp","Hm"), select="exip_pen", d4=True)) bit for bit.

  gate      "off" | "deterministic"   G1: Hp only if the strongest residual angle-delay cell exceeds tau_peak (x n0_hat);
                                      Hm only if the EXIP cost of the base localisation exceeds tau_cost        [DESIGN]
  recovery  "v3" | "targeted" | "combined"   v3: Hp, Hm; targeted: endfire AoA alias flips (+ optional pruning to three
                                      paths); combined: both                                                    [DESIGN]
  selector  "frozen_B4"               J = f/n0_hat + 0.5*EXIP cost + Z*K, eligibility and D4 exactly as v3 (no new selector
                                      was justified, docs/b2f3r_v31/r4_support_selection.md)
  runtime   "baseline" | "cached" | "bounded"   cached: exact reuse (residual peak shared by the gate and Hp; one-window
                                      refinement of the local search); bounded: additionally at most `max_hyps` generated
                                      hypotheses per observation (heuristic, changes outputs)

Inference inputs: y, pilots, beams, configuration, BS position; no N0, condition label or ground truth."""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..b2f3r_v2 import geometry_recovery as GR
from ..b2f3r_v3 import hypotheses as HY
from ..b2f3r_v3.candidate import B2f3Rv3, _cost, bound, score
from ..models.features import correlation_cube
from ..models.path_init import index_to_sine
from ..signal_processing import beamspace
from ..signal_processing.model import ObservationModel
from . import runtime_optimizations as RO
from . import targeted_recovery as TR

TAU_PEAK, TAU_COST, ENDFIRE = 18.0, 0.5, 0.95


def residual_peak(y, ctx0, e0):
    """(P_max, (aod, aoa)) of the residual angle-delay map: the frozen VL._residual_peak computation, also returning the
    peak statistic (cell energy / n0_hat because the correlation cube is normalised by sqrt(n0))."""
    model = ObservationModel(ctx0)
    r = y - model.mean(e0)
    c = correlation_cube(r, ctx0)
    N = c.shape[0]
    P = np.abs(np.fft.ifft(c, axis=0) * np.sqrt(N)) ** 2
    b, it, ir = np.unravel_index(int(np.argmax(P)), P.shape)
    cfg = ctx0.config
    e = cfg.estimator
    nt, nr = cfg.array.num_tx, cfg.array.num_rx
    pk = (float(beamspace.sine_to_angle(float(np.clip(index_to_sine(it, nt), -1, 1)), e.aod_branch)),
          float(beamspace.sine_to_angle(float(np.clip(index_to_sine(ir, nr), -1, 1)), e.aoa_branch)))
    return float(P[b, it, ir]), pk, r, model


def plus_one_reuse(y, ctx0, e0, r, model, peak):
    """Hp with the residual and residual peak already computed by the gate (identical to HY.plus_one)."""
    from ..b2f3r_v3.hypotheses import _refit
    from ..models.v3_aosage import _fit
    from ..signal_processing.model import ChannelEstimate
    centers = [(e0.aod[k], e0.aoa[k]) for k in range(e0.num_paths)] + [peak]
    _, tau, aod, aoa, g = RO.local_best(model, ctx0, r, centers)
    e = _refit(y, ctx0, ChannelEstimate(np.r_[e0.delay, tau], np.r_[e0.aod, aod], np.r_[e0.aoa, aoa], np.r_[e0.gain_tilde, g]))
    return e, _fit(model, y, e)


class B2f3Rv31(B2f3Rv3):
    requires_oracle = False

    def __init__(self, gate: str = "off", recovery: str = "v3", selector: str = "frozen_B4", runtime: str = "baseline",
                 tau_peak: float = TAU_PEAK, tau_cost: float = TAU_COST, endfire: float = ENDFIRE, max_flips: int = 2,
                 prune: bool = False, max_hyps: int | None = None, **kw):
        kw.setdefault("hyps", ("Hp", "Hm"))
        kw.setdefault("select", "exip_pen")
        kw.setdefault("d4", True)
        super().__init__(**kw)
        assert gate in ("off", "deterministic") and recovery in ("v3", "targeted", "combined")
        assert selector == "frozen_B4" and runtime in ("baseline", "cached", "bounded")
        self.gate, self.recovery, self.selector, self.runtime = gate, recovery, selector, runtime
        self.tau_peak, self.tau_cost, self.endfire, self.max_flips, self.prune = tau_peak, tau_cost, endfire, max_flips, prune
        self.max_hyps = max_hyps if runtime == "bounded" else None
        self.name = (f"b2f3r_v31[gate={gate}({tau_peak},{tau_cost}),recovery={recovery}(prune={int(prune)}),selector={selector},"
                     f"runtime={runtime}({self.max_hyps})]")

    def _all_off(self):
        return self.gate == "off" and self.recovery == "v3" and self.runtime == "baseline"

    def _hypotheses(self, out, base, h0, y, context):
        if self._all_off():
            return super()._hypotheses(out, base, h0, y, context)
        e0, f0, n00 = h0
        eb, fb, _ = base
        ctx0 = dataclasses.replace(context, n0=n00)
        kind = self.select
        s_base = score(kind, fb, eb.num_paths, n00, _cost(out))
        cands = [("base", s_base, out)]
        info = {"triggered": 1, "scores": {"base": float(s_base)}, "skipped": {}, "t": {}, "generated": [], "gate": {}}
        t0 = time.perf_counter()
        need_peak = self.gate == "deterministic" or self.runtime in ("cached", "bounded")
        pk = None
        if need_peak and self.recovery in ("v3", "combined"):
            pk = residual_peak(y, ctx0, e0)
        info["t"]["gate"] = time.perf_counter() - t0
        gens = []
        if self.recovery in ("v3", "combined"):
            run_hp = self.gate == "off" or pk[0] > self.tau_peak
            run_hm = self.gate == "off" or _cost(out) > self.tau_cost
            info["gate"] = {"peak": None if pk is None else pk[0], "cost": _cost(out), "Hp": int(run_hp), "Hm": int(run_hm)}
            if run_hp:
                if pk is not None and self.runtime in ("cached", "bounded"):
                    gens.append(("Hp", lambda: plus_one_reuse(y, ctx0, e0, pk[2], pk[3], pk[1])))
                else:
                    gens.append(("Hp", lambda: HY.plus_one(y, ctx0, e0)))
        if self.recovery in ("targeted", "combined"):
            for k in TR.endfire_paths(eb, "aoa", self.endfire)[: self.max_flips]:
                gens.append((f"flip_aoa{k}", lambda k=k: TR.alias_flip(y, ctx0, eb, k, "aoa")))
        if self.recovery in ("v3", "combined") and (self.gate == "off" or _cost(out) > self.tau_cost):
            gens.append(("Hm", lambda: HY.minus_one(y, ctx0, e0)))
        if self.recovery in ("targeted", "combined") and self.prune:
            gens.append(("prune3", lambda: TR.prune_to(y, ctx0, eb, 3)))
        if self.max_hyps is not None:
            gens = gens[: self.max_hyps]
        for h, fn in gens:
            t1 = time.perf_counter()
            res = fn()
            if res is None:
                continue
            e, f = res
            info["generated"].append(h)
            if HY.same_support(e, e0, context):
                info["skipped"][h] = "duplicate"
            elif self.d4 and bound(kind, f, e.num_paths, n00) >= s_base:
                info["skipped"][h] = "bound"
            else:
                o = GR.localize(e, y, ctx0, self.frozen.geom, self.geo)
                if o.info.get("c_hat") != "olos":
                    info["skipped"][h] = "not_olos"
                else:
                    s = score(kind, f, e.num_paths, n00, _cost(o))
                    info["scores"][h] = float(s)
                    cands.append((h, s, o))
            info["t"][h] = time.perf_counter() - t1
        order = sorted(range(len(cands)), key=lambda i: (cands[i][1], i))
        sel = cands[order[0]]
        margin = cands[order[1]][1] - sel[1] if len(order) > 1 else np.inf
        info.update({"selected": sel[0], "margin": float(margin), "ambiguous": int(sel[0] != "base" or margin < self.flag_theta),
                     "n_hyps": len(info["generated"]), "n_loc": len(cands) - 1})
        return sel[2], info

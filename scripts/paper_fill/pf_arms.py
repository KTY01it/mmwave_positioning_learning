"""PAPER-FILL experiment arms (new code; nothing in src/ or configs/ is modified).

Every arm is built from the frozen Selective Recovery estimator (C1 = B2f3RFinal(relprune_rho=15), arm C1_modelorder in
configs/experiments/b2f3r_final/arms.yaml) and differs from it in exactly the documented way.

B1 component ablations (class PFAblation; `_hypotheses` is a verbatim copy of b2f3r_final/candidate.py::B2f3RFinal._hypotheses
   with two switches that only filter which generators are appended):
   pf_full        all on                      -> must reproduce C1 bit for bit (asserted on the stored confirmation records)
   pf_nosupp      no support hypotheses       (H+, H-, H_prune not generated)
   pf_noendfire   no endfire hypotheses       (H_flip not generated)
   pf_none        neither                     (base estimate H_b only)
   pf_nogate      reliability gates removed   (r_peak > 18 and c_EXIP > 0.5 replaced by "always"; H+ and H- always generated
                                               for OLOS-routed outputs; the OLOS routing itself is kept, see report)
B3 one-at-a-time sensitivity (frozen kwargs of B2f3RFinal, except Z which is the module constant of the joint score):
   pf_z50 / pf_z85              Z of J(H) = f/N0 + c_EXIP/2 + Z K   (b2f3r_v3.candidate.Z patched during estimate only)
   pf_rp12 / pf_rp25            r_peak gate tau_peak
   pf_ce025 / pf_ce1            c_EXIP gate tau_cost
   pf_ef090 / pf_ef098          endfire threshold |sin theta_R|
   pf_pr10 / pf_pr20            relative pruning rho [dB]
B4 literature baselines in the same infrastructure (no competing hypotheses, one localisation):
   pf_filter      "filter instead of select" after Kakkavas et al. (IEEE TWC 2021): relative-gain pruning + single-bounce
                  triangle condition on H0, LS-gain + SAGE refit, then the frozen localisation (EXIP/LMA) once
   pf_pfa41       model order by the false-alarm threshold of Shahmansoori et al. (IEEE TWC 2018, Eq. (41), P_FA = 1e-3)
                  in place of Z in the path-acceptance / backward tests of the model-order loop, then the frozen localisation once
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
from scipy.special import gammaincinv

from positioning.b2f3r_final import model_order as MO
from positioning.b2f3r_final.candidate import B2f3RFinal
from positioning.b2f3r_v2 import geometry_recovery as GR
from positioning.b2f3r_v2 import path_recovery as PR
from positioning.b2f3r_v3 import candidate as V3C
from positioning.b2f3r_v3 import hypotheses as HY
from positioning.b2f3r_v3.candidate import _cost, bound, score
from positioning.b2f3r_v3.hypotheses import _refit
from positioning.b2f3r_v31 import targeted_recovery as TR
from positioning.b2f3r_v31.candidate import plus_one_reuse, residual_peak
from positioning.models.path_init import ls_gains
from positioning.models.v3_aosage import _fit
from positioning.robust import loop as RL
from positioning.signal_processing.model import ChannelEstimate, ObservationModel


class PFAblation(B2f3RFinal):
    """B2f3RFinal with switches on the hypothesis generators (defaults = C1)."""

    def __init__(self, use_support: bool = True, use_endfire: bool = True, **kw):
        kw.setdefault("relprune_rho", 15.0)
        super().__init__(**kw)
        self.use_support, self.use_endfire = use_support, use_endfire
        self.name = f"pf_ablation[support={use_support},endfire={use_endfire},tau_peak={self.tau_peak},tau_cost={self.tau_cost}]"

    def _hypotheses(self, out, base, h0, y, context):
        # verbatim copy of B2f3RFinal._hypotheses (src/positioning/b2f3r_final/candidate.py) + the two switches (marked ##)
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
        if self.use_support and pk[0] > self.tau_peak:                                                   ##
            gens.append(("Hp", lambda: plus_one_reuse(y, ctx0, e0, pk[2], pk[3], pk[1])))
        if self.use_endfire:                                                                             ##
            for k in TR.endfire_paths(eb, "aoa", self.endfire)[: self.max_flips]:
                gens.append((f"flip_aoa{k}", lambda k=k: TR.alias_flip(y, ctx0, eb, k, "aoa")))
        if self.use_support and _cost(out) > self.tau_cost:                                               ##
            gens.append(("Hm", lambda: HY.minus_one(y, ctx0, e0)))
        if self.use_support and self.relprune_rho is not None:                                            ##
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


class ZScore:
    """C1 with the Z of the joint score J(H) changed (module constant b2f3r_v3.candidate.Z, restored after every call).
    The model-order loop's own path test (b2f3r_v2.path_recovery.Z) is NOT changed."""

    requires_oracle = False

    def __init__(self, z):
        self.inner, self.z = PFAblation(), float(z)
        self.name = f"pf_zscore[{z}]"

    def estimate(self, y, context, gt=None):
        old = V3C.Z
        V3C.Z = self.z
        try:
            return self.inner.estimate(y, context)
        finally:
            V3C.Z = old


def single_bounce_ok(e, ref):
    """Triangle condition of a single-bounce path w.r.t. the reference (LOS-candidate) path: the AoD and AoA offsets have
    opposite signs, (theta_T,l - theta_T,ref) * (theta_R,l - theta_R,ref) < 0 (angles wrapped to (-pi, pi])."""
    w = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    dT, dR = w(e.aod - e.aod[ref]), w(e.aoa - e.aoa[ref])
    ok = dT * dR < 0
    ok[ref] = True
    return ok


class FilterBaseline(B2f3RFinal):
    """'Filter instead of select' (Kakkavas et al., TWC 2021): H0 of the frozen front end -> keep paths with
    E_k >= max E - rho_db (relative gain) that also satisfy the single-bounce triangle condition w.r.t. the earliest
    path (LOS candidate) -> LS gains + SAGE refit -> frozen localisation once. No G2 repair, no hypotheses."""

    def __init__(self, rho_db=15.0, k_min=2, **kw):
        super().__init__(relprune_rho=15.0, **kw)
        self.rho_db, self.k_min = rho_db, k_min
        self.name = f"pf_filter[rho={rho_db}]"

    def estimate(self, y, context, gt=None):
        t0 = time.perf_counter()
        from positioning.b2f3r import noise as NZ1
        if self.noise != "known":
            context = dataclasses.replace(context, n0=NZ1.plug_in(y, context, "E2"))
        e0, f0, n0, log = self.paths(y, context)
        ctx0 = dataclasses.replace(context, n0=n0)
        s = MO.path_snr_db(e0, ctx0)
        ref = int(np.argmin(e0.delay))
        keep = [k for k in range(e0.num_paths) if s[k] >= s.max() - self.rho_db and single_bounce_ok(e0, ref)[k]]
        filt = {"n0_paths": int(e0.num_paths), "kept": len(keep), "fallback": 0}
        e = e0
        if len(keep) < e0.num_paths:
            if len(keep) >= self.k_min:
                model = ObservationModel(ctx0)
                sub = e0.subset(sorted(keep))
                e = _refit(y, ctx0, ChannelEstimate(sub.delay, sub.aod, sub.aoa, ls_gains(model, y, sub.delay, sub.aod, sub.aoa)))
            else:
                filt["fallback"] = 1                       # fewer than k_min paths would remain: H0 is used unfiltered
        out = GR.localize(e, y, ctx0, self.frozen.geom, self.geo)
        out.info.update({"loop": log, "n0_used": float(n0), "filter": filt, "t_total": time.perf_counter() - t0})
        return out


def z_eq41(config, pfa=1e-3):
    """Eq. (41) of Shahmansoori et al. in the units of the loop's path test (explained energy / N0):
    delta / N0 = P^{-1}(N, (1 - P_FA)^{1/cols}), cols = Nt Nr * offgrid_threshold_factor (the frozen classical setting)."""
    e = config.estimator
    cols = config.array.num_tx * config.array.num_rx * e.offgrid_threshold_factor
    return float(gammaincinv(config.physics.num_subcarriers, (1.0 - pfa) ** (1.0 / cols)))


class PFA41Baseline(B2f3RFinal):
    """Frozen front end with the model-order thresholds Z (forward path acceptance, b2f3r_v2.path_recovery.Z, and backward
    test, robust.loop.Z) replaced by Eq. (41) at P_FA = 1e-3, then the frozen localisation once (no G2, no hypotheses)."""

    def __init__(self, pfa=1e-3, **kw):
        super().__init__(relprune_rho=15.0, **kw)
        self.pfa = pfa
        self.name = f"pf_pfa41[{pfa}]"

    def estimate(self, y, context, gt=None):
        t0 = time.perf_counter()
        from positioning.b2f3r import noise as NZ1
        zq = z_eq41(context.config, self.pfa)
        old = (PR.Z, RL.Z)
        PR.Z, RL.Z = zq, zq
        try:
            if self.noise != "known":
                context = dataclasses.replace(context, n0=NZ1.plug_in(y, context, "E2"))
            e0, f0, n0, log = self.paths(y, context)
        finally:
            PR.Z, RL.Z = old
        out = GR.localize(e0, y, dataclasses.replace(context, n0=n0), self.frozen.geom, self.geo)
        out.info.update({"loop": log, "n0_used": float(n0), "z_eq41": zq, "t_total": time.perf_counter() - t0})
        return out


def build(name):
    inf = float("inf")
    table = {
        "pf_full": lambda: PFAblation(),
        "pf_nosupp": lambda: PFAblation(use_support=False),
        "pf_noendfire": lambda: PFAblation(use_endfire=False),
        "pf_none": lambda: PFAblation(use_support=False, use_endfire=False),
        "pf_nogate": lambda: PFAblation(tau_peak=-inf, tau_cost=-inf),
        "pf_z50": lambda: ZScore(50.0), "pf_z85": lambda: ZScore(85.0),
        "pf_rp12": lambda: PFAblation(tau_peak=12.0), "pf_rp25": lambda: PFAblation(tau_peak=25.0),
        "pf_ce025": lambda: PFAblation(tau_cost=0.25), "pf_ce1": lambda: PFAblation(tau_cost=1.0),
        "pf_ef090": lambda: PFAblation(endfire=0.90), "pf_ef098": lambda: PFAblation(endfire=0.98),
        "pf_pr10": lambda: PFAblation(relprune_rho=10.0), "pf_pr20": lambda: PFAblation(relprune_rho=20.0),
        "pf_filter": lambda: FilterBaseline(),
        "pf_pfa41": lambda: PFA41Baseline(),
    }
    return table[name]()

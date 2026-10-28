"""T2: OLOS geometry stage on a given B2f3 channel estimate (paired substitution of the initial point).

Every variant uses the same paths, the same weight matrix and the same EXIP/LMA (localization.exip_lma.exip_lma, run
to convergence with its frozen rules). Only the starting point(s) differ:

  learned   frozen v1R GeomInitNet (what branch_localize does)
  cls1      frozen classical alpha grid, Eqs. (64)-(65), LMA from the best initial cost (classical_single setting)
  cls3      the three best initial costs
  clsall    LMA from every grid trial (strong classical setting)
A multi-start keeps the result with the smallest EXIP cost among the union of its starts (observation-only).
"""
from __future__ import annotations

import time

import numpy as np

from ..localization.exip_lma import exip_lma, pack, weight_matrix
from ..localization.olos import olos_localize
from ..models.base import PositionEstimate
from ..models.router import branch_localize, classical_route
from ..signal_processing.model import ObservationModel

VARIANTS = ("learned", "cls1", "cls3", "clsall")


def olos_fits(est, context, geom, variants=VARIANTS):
    """{variant: (EXIPResult, seconds)} for an estimate routed to OLOS (K >= 2). `est` in any order."""
    cfg = context.config
    e = est.sorted_by_delay()
    c = cfg.physics.speed_of_light
    w = cfg.estimator.exip_weighting
    out = {}
    for v in variants:
        t0 = time.perf_counter()
        if v == "learned":
            p0, a0, s0 = geom.predict(e, c)
            r = exip_lma(e, weight_matrix(e, context, w), pack(p0, a0, s0, e.gain_tilde, False), False,
                         context.bs_position, c)
        else:
            k = {"cls1": 1, "cls3": 3, "clsall": None}[v]
            r = olos_localize(e, context, w, cfg.estimator.olos_alpha_step, cfg.estimator.olos_alpha_max,
                              pairs=cfg.estimator.olos_pairs, top_k=k)
        out[v] = (r, time.perf_counter() - t0)
    return e, out


def best_of(fits, names):
    """Smallest EXIP cost among the named fits (None results and non-finite costs lose)."""
    ok = [(fits[n][0].cost if fits[n][0] is not None and np.isfinite(fits[n][0].cost) else np.inf, i, n)
          for i, n in enumerate(names)]
    return min(ok)[2]


def to_estimate(e, r, y, context) -> PositionEstimate:
    """PositionEstimate of an OLOS fit with the same gate record as models.router.branch_localize."""
    out = PositionEstimate(r.position, r.orientation, "olos", r.scatterers, e, r.cost,
                           info={"lma_nfev": r.nfev, "lma_success": r.success})
    M = y.size
    res = float(np.sum(np.abs(y - ObservationModel(context).mean(e)) ** 2) / context.n0)
    out.info["gate"] = {"r_norm": res / M, "z": (res - M) / np.sqrt(M), "v": float(r.cost) if np.isfinite(r.cost) else 0.0}
    return out


C0 = 1.0          # cost margin of rule S1, in units of the FIM-weighted EXIP cost (chi-square scale)


def localize(est, y, context, geom, policy: str = "S1") -> PositionEstimate:
    """Frozen condition test, then the geometry stage.

    policy "frozen": models.router.branch_localize, unchanged.
    policy "S1" (selected on t2_dev, scripts/90_t2_geometry.py): in the OLOS branch the learned start is used as in the
    frozen method; only if its converged EXIP cost exceeds C0 is the classical alpha-grid start (best initial cost,
    one LMA) also run, and it replaces the learned fit only if its cost is lower by more than C0. Ties keep the
    learned fit. LOS and NLOS branches are the frozen ones."""
    cond, _ = classical_route(est, context)
    if policy == "frozen" or cond != "olos" or est.num_paths < 2:
        out = branch_localize(est, y, context, cond, geom)
        out.info["geo"] = "frozen"
    else:
        e, fits = olos_fits(est, context, geom, ("learned",))
        r, used = fits["learned"][0], "learned"
        if not np.isfinite(r.cost) or r.cost > C0:
            rc = olos_fits(est, context, geom, ("cls1",))[1]["cls1"][0]
            if rc is not None and np.isfinite(rc.cost) and (not np.isfinite(r.cost) or r.cost - rc.cost > C0):
                r, used = rc, "cls1"
        out = to_estimate(e, r, y, context)
        out.info["geo"] = used
    out.info["c_hat"] = cond
    return out

"""T2 (v2): geometry and support recovery for the OLOS branch.

  G1  observation-based choice of the geometric start: rule S1 of B2f3R-v1 (positioning.b2f3r.geometry.localize),
      reused unchanged.
  G2  selective multipath-support repair: when the final fit of the routed estimate is inconsistent, the path stage is
      re-run from the classical coarse support (robust.lowg.cf_hybrid, the support source of A2b) and the result with
      the better penalised data fit is kept. Trigger and choice are defined in `support_trigger` / `choose`. [DESIGN]
"""
from __future__ import annotations

import numpy as np

from ..b2f3r import geometry as GEO1
from ..models.v3_aosage import Z_DELTA

Z = Z_DELTA[1e-8]
localize = GEO1.localize
C0 = GEO1.C0


def support_trigger(out) -> bool:
    """G2 trigger (selected on v2dev_g2b by the rule in outputs/b2f3r_v2/selection_rules.json): the estimate was routed
    to OLOS and the OLOS geometric fit is inconsistent (EXIP cost above C0, or not finite)."""
    return out.info.get("c_hat") == "olos" and not (np.isfinite(out.cost) and out.cost <= C0)


def choose(f_b, k_b, f_h, k_h, n0) -> bool:
    """True if the repaired support explains the observation better after a cost of z per path (the frozen path-test
    threshold), i.e. the comparison the model-order loop itself uses to accept a path."""
    return f_h / n0 + Z * k_h < f_b / n0 + Z * k_b


def support_repair(cand, out, est, f, n0, y, context, log):
    """G2: re-run the path stage from the classical coarse support and keep it only if `choose` prefers it."""
    if not support_trigger(out):
        return out, 0
    import dataclasses
    est_h, f_h, n0_h, _ = cand.paths(y, context, force_coarse=True)
    if not choose(f, est.num_paths, f_h, est_h.num_paths, n0):
        return out, 1
    out_h = localize(est_h, y, dataclasses.replace(context, n0=n0_h), cand.frozen.geom, cand.geo)
    return out_h, 2

"""Unknown propagation condition (Sec. IV-D): solve under NLOS and OLOS, then decide.

rule "min_cost" (paper): keep the hypothesis with the smaller cost (63) vs (64).
rule "chi2" [DESIGN]: with W = J_eta the EXIP cost of the correct hypothesis is asymptotically
chi-square with dof = 5K - #unknowns (NLOS: K-1, OLOS: K-3). NLOS (LOS geometry, more constrained) is
kept unless rejected at level ``test_level``; with K = 3 the OLOS fit is exact (dof 0, v_olos ~ 0) so the
paper's rule cannot discriminate.
"""
from __future__ import annotations

from scipy.stats import chi2

from ..signal_processing.model import ChannelEstimate
from .nlos import nlos_localize
from .olos import olos_localize


def unknown_condition_localize(est: ChannelEstimate, context, weighting="fim", alpha_step=0.05, alpha_max=0.5,
                               top_k=None, rule: str = "chi2", test_level: float = 0.01,
                               los_selection: str = "min_delay", prune_level: float | None = None,
                               pairs: str = "all"):
    K = est.num_paths
    # the hypothesis test uses the unpruned NLOS fit: pruning would remove paths until the NLOS model fits
    # (down to LOS only, v_nlos = 0) and so defeat the test; pruning is applied to the final NLOS estimate only
    nl = nlos_localize(est, context, weighting, los_selection, None)
    ol = olos_localize(est, context, weighting, alpha_step, alpha_max, pairs=pairs, top_k=top_k) if K >= 2 else None
    dof_nlos, dof_olos = K - 1, K - 3
    p_nlos = float(chi2.sf(nl.cost, dof_nlos)) if dof_nlos > 0 else 1.0
    if ol is None:
        chosen, cond = nl, "nlos"
    elif rule == "min_cost":
        chosen, cond = (nl, "nlos") if nl.cost <= ol.cost else (ol, "olos")
    elif rule == "chi2":
        chosen, cond = (nl, "nlos") if p_nlos >= test_level else (ol, "olos")
    else:
        raise ValueError(f"unknown rule {rule!r}")
    if cond == "nlos" and prune_level is not None:
        chosen = nlos_localize(est, context, weighting, los_selection, prune_level)
    ratio = nl.cost / ol.cost if ol is not None and ol.cost > 0 else float("inf")
    return chosen, cond, {"v_nlos": nl.cost, "v_olos": None if ol is None else ol.cost, "delta_v": ratio,
                          "dof_nlos": dof_nlos, "dof_olos": dof_olos, "p_nlos": p_nlos, "nlos": nl, "olos": ol}

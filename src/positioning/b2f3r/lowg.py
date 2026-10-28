"""T1: low-G support repair. Path stages (frozen support / classical coarse support / cyclic-search negative control)
on the verified copy of the B2f3 loop (positioning.robust.loop), and observable reliability features of the frozen
stage. Nothing here reads ground truth, the propagation condition or the true K.

  frozen   network candidates (identical to V3Candidate; unit-tested)
  hybrid   A1: support from the classical 4x off-grid DCS-SOMP coarse stage, then the unchanged loop
  cgs      negative control: cyclic global AoD search + post-loop pass (candidate L2 of the earlier program)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..models.path_init import ls_gains, network_input
from ..models.v3_aosage import _fit
from ..robust import loop as LP
from ..robust import lowg as LG
from ..signal_processing.model import ChannelEstimate, ObservationModel

G_LOW = 10


def path_stage(cand, y, context, mode: str = "frozen"):
    """(est, f, log, captured network candidates) of one path stage. `cand` is a frozen V3Candidate."""
    log, cap = {}, {}

    def cf(conf, pool, y_, c_, model, out):
        cap["conf"], cap["pool"] = list(conf), list(pool)
        if mode == "hybrid":
            return LG.cf_hybrid(conf, pool, y_, c_, model, out)
        if mode == "cgs":
            return LG.cf_cgs(conf, pool, y_, c_, model, out)
        return conf, pool
    est, f = LP.model_order_loop(cand.pe, y, context, network_input(y, context), it_mid=cand.it_mid, cand_fn=cf,
                                 post_fn=LG.post_cyclic if mode == "cgs" else None, log=log)
    return est, f, log, cap


def lobe_ratios(est, y, context):
    """Per path: (largest single-path likelihood over the whole AoD range outside the path's own cell) / (likelihood in
    its own cell), on the residual of the other estimated paths. > 1: another transmit lobe explains the data better."""
    cfg = context.config
    model = ObservationModel(context)
    sc = LG.AodScanner(model, cfg)
    sg = np.sin(sc.grid)
    out = []
    for k in range(est.num_paths):
        idx = [j for j in range(est.num_paths) if j != k]
        r = y if not idx else y - model.mean(est.subset(idx))
        L = sc.scan(r, est.delay[k], est.aoa[k])
        near = np.abs(sg - np.sin(est.aod[k])) <= 2.0 / cfg.array.num_tx
        cur = float(L[near].max()) if near.any() else 0.0
        out.append(float(L[~near].max() / max(cur, 1e-300)))
    return out


def features(est, f, log, cap, y, context) -> dict:
    """Observable reliability features of a finished path stage."""
    M, n0 = y.size, context.n0
    eps = 0.5 * chi2.ppf(1 - LP.PFA, 2 * M)
    lr = lobe_ratios(est, y, context)
    conf, pool = cap.get("conf", []), cap.get("pool", [])
    return {"K": int(est.num_paths), "n_conf": len(conf), "n_pool": len(pool),
            "p_min_conf": float(min((c[0] for c in conf), default=0.0)), "p_max_pool": float(max((c[0] for c in pool), default=0.0)),
            "fwd": int(log["fwd_entered"]), "acc": int(log["accepted"]), "rem": int(log["removed"]), "rounds": int(log["rounds"]),
            "z_res": float((f / n0 - M) / np.sqrt(M)), "stop_fail": int(f / n0 >= eps),
            "lobe_max": float(max(lr)), "lobe_n": int(sum(v > 1.0 for v in lr)), "f": float(f)}

"""R2: candidates and baselines for the initial support at low beam counts (docs/robustness_completion_plan.md, Sec. 3).

All functions use only (y, context); context.n0 is the estimated noise variance. The frozen path-test threshold is
used wherever a path is tested. Nothing here reads ground truth, the propagation condition or the true K.

  L1      cyclic global AoD search with successive residual cancellation on the initial support
  L2      L1 + one cyclic global re-evaluation of the final model after the model-order loop
  rescan  baseline: one global AoD rescan of every initial path (root-cause audit, arm R2)
  hybrid  baseline: the classical 4x coarse stage supplies the initial support
"""
from __future__ import annotations

import numpy as np

from ..models.path_init import ls_gains
from ..models.v3_aosage import _fit
from ..signal_processing.coarse import coarse_estimate
from ..signal_processing.model import ChannelEstimate
from . import loop as LP

OS = 4
MAX_PASSES = 3
G_LOW = 10


class AodScanner:
    """Single-path concentrated likelihood over the whole AoD range (4x grid, front branch) at a given delay and AoA.
    Same statistic as signal_processing.sage.objective_aod, with the Tx projections of the grid computed once."""

    def __init__(self, model, cfg):
        nt = cfg.array.num_tx
        self.model, self.nt = model, nt
        self.grid = np.arcsin(np.clip(np.linspace(-1, 1, OS * nt, endpoint=False) + 1.0 / (OS * nt), -1, 1))
        self.tp = model.tx_projection(self.grid)                         # [C, N, G]
        self.den = np.sum(np.abs(self.tp) ** 2, axis=(1, 2))

    def scan(self, r, tau, aoa):
        e = self.model.delay_phasor(tau)
        W = np.conj(e)[:, None] * np.einsum("nr,ngr->ng", self.model.rx_response(aoa).conj(), r)
        return np.abs(np.einsum("cng,ng->c", self.tp.conj(), W)) ** 2 / self.den

    def best(self, r, tau, aoa):
        return float(self.grid[int(np.argmax(self.scan(r, tau, aoa)))])


def _est(model, y, cur):
    d, a, b = (np.array([c[q] for c in cur]) for q in (3, 1, 2))
    return ChannelEstimate(d, a, b, ls_gains(model, y, d, a, b))


def _residual(model, y, cur):
    return y if not cur else y - model.mean(_est(model, y, cur))


def _same_cell(c, k, Ts, nt, nr, aod=True):
    return (abs(c[3] - k[3]) <= Ts and abs(np.sin(c[2]) - np.sin(k[2])) <= 2.0 / nr and
            (not aod or abs(np.sin(c[1]) - np.sin(k[1])) <= 2.0 / nt))


def _merge(cur, Ts, nt, nr):
    keep = []
    for c in cur:
        if not any(_same_cell(c, k, Ts, nt, nr) for k in keep):
            keep.append(c)
    return keep


def cf_rescan(conf, pool, y, context, model, out):
    """Baseline: one pass, AoD of every initial path := global maximiser on the residual of the other initial paths."""
    cfg = context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    sc = AodScanner(model, cfg)
    cur = [list(c) for c in conf]
    for i in range(len(cur)):
        cur[i][1] = sc.best(_residual(model, y, [c for j, c in enumerate(cur) if j != i]), cur[i][3], cur[i][2])
    return [tuple(c) for c in _merge(cur, Ts, nt, nr)], pool


def cf_hybrid(conf, pool, y, context, model, out):
    """Baseline: initial support from the classical 4x off-grid DCS-SOMP coarse stage (threshold (41) with context.n0)."""
    ce = coarse_estimate(y, context)
    if ce.num_paths == 0:
        return conf, pool
    return [(1.0, float(a), float(b), float(t)) for t, a, b in zip(ce.delay, ce.aod, ce.aoa)], []


def cf_cgs(conf, pool, y, context, model, out):
    """L1: cyclic global AoD search with successive residual cancellation.

    Slots = network candidates grouped by (delay cell, AoA cell); a slot is initial if any member is confident.
    Pass 0: slots in order of probability, AoD = global maximiser on the residual of the slots already placed.
    Cyclic passes (<= 3): every slot is rescanned on the residual of all the others (gains re-solved jointly); slots in
    one cell are merged; every slot is re-tested with the frozen backward path test. Stop when no AoD moved by more than
    one cell. Pool slots are scanned once on the final residual."""
    cfg = context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    n0 = context.n0
    sc = AodScanner(model, cfg)
    nconf = len(conf)
    slots = []                                                            # [prob, aod, aoa, tau, is_conf]
    for i, c in enumerate(sorted(conf + pool, key=lambda q: -q[0])):
        hit = next((s for s in slots if _same_cell(c, s, Ts, nt, nr, aod=False)), None)
        is_conf = c in conf
        if hit is None:
            slots.append([c[0], c[1], c[2], c[3], is_conf])
        else:
            hit[4] = hit[4] or is_conf
    init = [s for s in slots if s[4]] or slots[:1]
    rest = [s for s in slots if s not in init]
    cur = []
    for s in init:                                                        # pass 0: successive cancellation
        cur.append([s[0], sc.best(_residual(model, y, cur), s[3], s[2]), s[2], s[3]])
    for _ in range(MAX_PASSES):
        moved = False
        for i in range(len(cur)):
            a = sc.best(_residual(model, y, [c for j, c in enumerate(cur) if j != i]), cur[i][3], cur[i][2])
            if abs(np.sin(a) - np.sin(cur[i][1])) > 2.0 / nt:
                moved = True
            cur[i][1] = a
        cur = _merge(cur, Ts, nt, nr)
        if len(cur) > 1:                                                  # re-test every slot (alias paths)
            est, changed = LP.backward_prune(model, y, _est(model, y, cur), n0)
            if changed:
                cur = [[1.0, float(a), float(b), float(t)] for t, a, b in zip(est.delay, est.aod, est.aoa)]
                moved = True
        if not moved:
            break
    r = _residual(model, y, cur)
    pool2 = [(s[0], sc.best(r, s[3], s[2]), s[2], s[3]) for s in rest]
    pool2 = [c for c in pool2 if not any(_same_cell(c, k, Ts, nt, nr) for k in cur)]
    return [(1.0, c[1], c[2], c[3]) for c in cur], pool2


def post_cyclic(est, f, y, context, model, refit):
    """L2: one cyclic global AoD re-evaluation of the final model, the frozen backward test, a refit. The new model is
    kept only if it explains the observation at least as well (no threshold)."""
    cfg = context.config
    nt = cfg.array.num_tx
    sc = AodScanner(model, cfg)
    aod = est.aod.copy()
    moved = False
    for i in range(est.num_paths):
        oth = [j for j in range(est.num_paths) if j != i]
        cur = ChannelEstimate(est.delay[oth], aod[oth], est.aoa[oth], ls_gains(model, y, est.delay[oth], aod[oth],
                                                                              est.aoa[oth])) if oth else None
        r = y if cur is None else y - model.mean(cur)
        a = sc.best(r, est.delay[i], est.aoa[i])
        if abs(np.sin(a) - np.sin(aod[i])) > 2.0 / nt:
            aod[i] = a
            moved = True
    if not moved:
        return est, f
    new = ChannelEstimate(est.delay, aod, est.aoa, ls_gains(model, y, est.delay, aod, est.aoa))
    new, _ = LP.backward_prune(model, y, new, context.n0)
    new = refit(ChannelEstimate(new.delay, new.aod, new.aoa, ls_gains(model, y, new.delay, new.aod, new.aoa)))
    f_new = _fit(model, y, new)
    return (new, f_new) if f_new <= f else (est, f)


VARIANTS = {"L1": dict(cand_fn=cf_cgs), "L2": dict(cand_fn=cf_cgs, post_fn=post_cyclic),
            "rescan": dict(cand_fn=cf_rescan), "hybrid": dict(cand_fn=cf_hybrid)}

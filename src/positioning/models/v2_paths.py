"""V2-A (docs/v2_protocol.md): multi-hypothesis physics-guided path estimation on top of the learned path heatmap.

Front end: the 6A path-initialiser network retrained on train_v2 with its original recipe (= v1R), unchanged.
New stage (replaces "single decode + hard (41) prune"):
  1. Candidate set with existence probabilities: every 3x3 local maximum of the objectness map above a low floor,
     up to N_CAND (set prediction with existence probabilities; cf. Carion et al., ECCV 2020 [LIT, principle]).
  2. AoD-lobe alternatives [DESIGN]: for each candidate a 1-D physical scan over AoD (4x oversampled grid) of the
     matched-filter statistic with its (tau, theta_Rx) fixed; the global maximiser is the second AoD hypothesis.
     (Diagnosis 2026-10-09: out-of-geometry failures of the learned path init are wrong AoD lobes with correct AoA and
     delay.)
  3. Hypotheses H(K, variant): the K most probable candidates (K = 1..K_MAX) with network AoDs or scanned AoDs.
     Gains by least squares; selection by the Bayesian information criterion with known noise,
     BIC = 2 ||y - y_hat||^2 / N0 + nu ln(2M), nu = 5K real parameters, M complex observations
     (Stoica & Selen, IEEE SPM 21(4), 2004 [LIT]). (41) is no longer a hard delete.
  4. One SAGE iteration on the selected hypothesis (as in 6A).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage
from .path_init import PathInitEstimator, decode, network_input

N_CAND, K_MAX, FLOOR, OS, COMPLETION_TRIES = 6, 4, 0.05, 4, 1


def _aod_scan(model: ObservationModel, y, tau, aoa, grid, tx_proj):
    """|<b(tau, theta, aoa), y>|^2 / ||b||^2 over the AoD grid (tx_proj [C, N, G] precomputed)."""
    z = np.einsum("ngr,nr->ng", y, model.rx_response(aoa).conj()) * model.delay_phasor(tau).conj()[:, None]
    num = np.abs(np.einsum("cng,ng->c", tx_proj.conj(), z)) ** 2
    den = np.sum(np.abs(tx_proj) ** 2, axis=(1, 2))           # the Rx / delay factors are common to all AoD trials
    return num / np.maximum(den, 1e-30)


def _peak(grid, sc):
    """Global maximiser with parabolic sub-grid interpolation."""
    j = int(np.argmax(sc))
    if 0 < j < len(sc) - 1:
        y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        return float(np.interp(j + np.clip(off, -0.5, 0.5), np.arange(len(grid)), grid))
    return float(grid[j])


def _dedupe(paths, cfg):
    """Merge candidates that fall in the same resolution cell (delay Ts, AoD 2/Nt, AoA 2/Nr in sine); keep the most
    probable one (paths are sorted by existence probability)."""
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    keep = []
    for c in paths:
        if all(abs(c[3] - k[3]) * 1.0 > 1.0 * Ts or abs(np.sin(c[1]) - np.sin(k[1])) > 2.0 / nt
               or abs(np.sin(c[2]) - np.sin(k[2])) > 2.0 / nr for k in keep):
            keep.append(c)
    return keep


def _bic_fit(model, y, paths, n0):
    """Least-squares gains and BIC (known noise) for a list of (prob, aod, aoa, tau_seconds)."""
    B = np.stack([model.path_basis(c[3], c[1], c[2]).ravel() for c in paths], axis=1)
    yv = y.ravel()
    g, *_ = np.linalg.lstsq(B, yv, rcond=None)
    r = yv - B @ g
    rss = float(np.real(np.vdot(r, r)))
    return 2 * rss / n0 + 5 * len(paths) * np.log(2 * y.size), g, r.reshape(y.shape)


def _residual_candidate(model, r, context, paths, grid, tx_proj, rank: int = 0):
    """Physics-guided completion [DESIGN]: strongest cell of the residual angle-delay map (unitary 1x grid, as the
    network input) outside the resolution cells of the current paths, then a fine AoD scan at its (tau, AoA)."""
    from ..signal_processing.beamspace import sine_to_angle
    from .features import correlation_cube
    from .path_init import index_to_sine
    cfg = context.config
    c = correlation_cube(r, context)                              # [N, Nt, Nr]
    N = c.shape[0]
    P = np.abs(np.fft.ifft(c, axis=0) * np.sqrt(N)) ** 2          # [delay bin, Nt, Nr]
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    for (_, aod, aoa, tau) in paths:                              # mask the cells of the paths already in the model
        b = int(round(tau / Ts)) % N
        it = int(round(np.sin(aod) * nt / 2 + (nt - 1) / 2))
        ir = int(round(np.sin(aoa) * nr / 2 + (nr - 1) / 2))
        P[max(b - 1, 0):b + 2, max(it - 2, 0):it + 3, max(ir - 2, 0):ir + 3] = 0
    for _ in range(rank):                                         # rank-th strongest cell (mask the stronger ones)
        b, it, ir = np.unravel_index(int(np.argmax(P)), P.shape)
        P[max(b - 1, 0):b + 2, max(it - 2, 0):it + 3, max(ir - 2, 0):ir + 3] = 0
    b, it, ir = np.unravel_index(int(np.argmax(P)), P.shape)
    e = cfg.estimator
    aoa = float(sine_to_angle(float(np.clip(index_to_sine(ir, nr), -1, 1)), e.aoa_branch))
    tau = b * Ts
    aod = _peak(grid, _aod_scan(model, r, tau, aoa, grid, tx_proj))
    return (0.0, aod, aoa, tau)


def _delta(context):
    """Detection threshold of the frozen baseline (Eq. (41), off-grid effective columns), as used by (41)-validation."""
    from ..signal_processing.dcs_somp import detection_threshold
    cfg = context.config
    e = cfg.estimator
    return detection_threshold(context.n0, cfg.physics.num_subcarriers,
                               cfg.array.num_tx * cfg.array.num_rx * e.offgrid_threshold_factor, e.pfa)


def _explained_energy_gain(y, context, est, k):
    """Energy explained by path k given the others: the statistic of (41) (per-subcarrier LS projection, same atoms
    as DCS-SOMP and validate_paths)."""
    from ..signal_processing import beamspace
    from ..signal_processing.coarse import _atom, _project
    model = ObservationModel(context)
    yb = beamspace.stack_observation(y).astype(complex)
    atoms = [_atom(model, est.aod[i], est.aoa[i]) for i in range(est.num_paths)]
    r_all = np.sum(np.abs(_project(yb, atoms)[1]) ** 2)
    r_wo = np.sum(np.abs(_project(yb, atoms[:k] + atoms[k + 1:])[1]) ** 2)
    return float(r_wo - r_all)


def _ls_residual(model, y, paths):
    if not paths:
        return y, np.zeros(0, complex)
    B = np.stack([model.path_basis(c[3], c[1], c[2]).ravel() for c in paths], axis=1)
    g, *_ = np.linalg.lstsq(B, y.ravel(), rcond=None)
    return (y.ravel() - B @ g).reshape(y.shape), g


def multi_hypothesis(pe: PathInitEstimator, x, y, context, completion_steps: int = 2):
    """Sequential physics-guided selection [DESIGN] (BIC on un-refined parameters over-selects K under off-grid /
    beam-squint mismatch, development log 2026-10-09): candidates by decreasing existence probability; each one gets
    the AoD maximising the single-path statistic on the residual of the already accepted paths (lobe correction) and is
    accepted iff its explained energy passes the frozen detection test (41); 1 SAGE iteration; then up to
    ``completion_steps`` missed-path searches on the refined residual, each accepted only by (41)."""
    cfg = context.config
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    t_net = time.perf_counter()
    Ts = cfg.physics.sampling_period
    model = ObservationModel(context)
    nt = cfg.array.num_tx
    grid = np.arcsin(np.clip(np.linspace(-1, 1, OS * nt, endpoint=False) + 1.0 / (OS * nt), -1, 1))
    tx_proj = model.tx_projection(grid)                          # [C, N, G]
    cand = _dedupe([(p, aod, aoa, tau * Ts) for p, aod, aoa, tau in decode(out, cfg, FLOOR, N_CAND)], cfg)
    delta = _delta(context)
    accepted = []
    for c in cand:
        if len(accepted) >= K_MAX:
            break
        r, _ = _ls_residual(model, y, accepted)
        c2 = (c[0], _peak(grid, _aod_scan(model, r, c[3], c[2], grid, tx_proj)), c[2], c[3])
        if any(abs(c2[3] - k[3]) <= Ts and abs(np.sin(c2[1]) - np.sin(k[1])) <= 2.0 / nt
               and abs(np.sin(c2[2]) - np.sin(k[2])) <= 2.0 / cfg.array.num_rx for k in accepted):
            continue                                              # collapsed onto an accepted path
        trial = accepted + [c2]
        est_t = ChannelEstimate(np.array([q[3] for q in trial]), np.array([q[1] for q in trial]),
                                np.array([q[2] for q in trial]), np.zeros(len(trial), complex))
        if _explained_energy_gain(y, context, est_t, len(trial) - 1) >= delta:
            accepted = trial
    if not accepted:                                              # at least one path (min_paths = 1, F22)
        c = cand[0]
        accepted = [(c[0], _peak(grid, _aod_scan(model, y, c[3], c[2], grid, tx_proj)), c[2], c[3])]
    t_scan = time.perf_counter()
    _, g = _ls_residual(model, y, accepted)
    init = ChannelEstimate(np.array([c[3] for c in accepted]), np.array([c[1] for c in accepted]),
                           np.array([c[2] for c in accepted]), g, info={"K": len(accepted)})
    est = sage(y, context, init, max_iter=1, tol=cfg.estimator.sage_tol).estimate
    t_sel = time.perf_counter()
    added = 0
    for _ in range(completion_steps):            # missed-path completion on the REFINED residual, test (41)
        if est.num_paths >= K_MAX:
            break
        r = y - model.mean(est)
        cur = [(0.0, est.aod[k], est.aoa[k], est.delay[k]) for k in range(est.num_paths)]
        trial = None
        for rank in range(COMPLETION_TRIES):                      # up to 3 residual cells per completion step
            c = _residual_candidate(model, r, context, cur, grid, tx_proj, rank)
            t_ = ChannelEstimate(np.r_[est.delay, c[3]], np.r_[est.aod, c[1]], np.r_[est.aoa, c[2]],
                                 np.r_[est.gain_tilde, 0.0])
            if _explained_energy_gain(y, context, t_, est.num_paths) >= delta:
                trial = t_
                break
        if trial is None:
            break
        _, g2 = _ls_residual(model, y, [(0, trial.aod[k], trial.aoa[k], trial.delay[k]) for k in range(trial.num_paths)])
        est = sage(y, context, ChannelEstimate(trial.delay, trial.aod, trial.aoa, g2), max_iter=1,
                   tol=cfg.estimator.sage_tol).estimate
        added += 1
    t_sage = time.perf_counter()
    est.info.update({**init.info, "added": added})
    return est, {"path_net": t_net - t0, "selection": t_scan - t_net, "sage": t_sel - t_scan, "completion": t_sage - t_sel}


def make(name):
    """Path estimator factory for scripts/31_v2_dev.py: f(y, context) -> (ChannelEstimate, stage times)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    if name == "v2a":
        pe = PathInitEstimator(str(root / "outputs" / "models" / "v1r" / "path_init" / "model.pt"), sage_iterations=1)

        def f(y, context):
            t0 = time.perf_counter()
            x = network_input(y, context)
            est, st = multi_hypothesis(pe, x, y, context)
            st["features"] = time.perf_counter() - t0 - sum(st.values())
            return est, st
        return f
    raise KeyError(name)

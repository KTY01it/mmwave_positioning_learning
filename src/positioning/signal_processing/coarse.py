"""Step 1 of the classical estimator (Sec. IV-B): DCS-SOMP -> AoD/AoA -> delay (54) and gain (53).

detection = "dcs_somp"          Algorithm 1 literally: grid atoms, residual from grid atoms.
detection = "dcs_somp_offgrid"  same atom selection (42) and threshold (41), but each selected atom is
                                refined to continuous (theta_Tx, theta_Rx) by maximising the same
                                per-subcarrier projected energy before Gram-Schmidt (46)-(47). Without this,
                                the leakage of one off-grid path into neighbouring atoms exceeds delta at
                                medium/high SNR and K is over-estimated (F17).
"""
from __future__ import annotations

import numpy as np

from . import beamspace, delay_estimation
from .dcs_somp import DCSSOMPResult, dcs_somp, detection_threshold
from .model import ChannelEstimate, ObservationModel
from .sage import angle_resolution, maximize_1d
from ..system.arrays import ula_response


def coupling_factor(Ut_col, Ur_col, aod, aoa, cfg) -> complex:
    """z^{(k)} of Eq. (50) evaluated at lambda_c (Eq. 52): u_Rx^H a_Rx(aoa) a_Tx^H(aod) u_Tx.

    Divided by ||u_Rx||^2 ||u_Tx||^2 so that the DCS-SOMP least-squares coefficient maps to h~ also for
    non-normalised dictionaries (main.m); this is exactly Eq. (50) for the paper's unitary dictionary.
    """
    lam, d, o = cfg.physics.wavelength_c, cfg.antenna_spacing, cfg.array.index_origin
    a_r = ula_response(aoa, cfg.array.num_rx, lam, d, o)
    a_t = ula_response(aod, cfg.array.num_tx, lam, d, o)
    scale = np.vdot(Ur_col, Ur_col).real * np.vdot(Ut_col, Ut_col).real
    return complex(np.vdot(Ur_col, a_r) * np.vdot(a_t, Ut_col) / scale)


# --------------------------------------------------------------------------------------
# off-grid atoms
# --------------------------------------------------------------------------------------
def _atom(model: ObservationModel, aod, aoa) -> np.ndarray:
    """Per-subcarrier atom b_n[g, r] = (a_Tx,n^H s_g[n]) a_Rx,n[r], flattened to [N, G*Nr]."""
    return (model.tx_projection(aod)[:, :, None] * model.rx_response(aoa)[:, None, :]).reshape(model.N, -1)


def _refine_atom(model: ObservationModel, R, aod, aoa, hw_t, hw_r, sweeps=3):
    """Maximise sum_n |b_n^H r_n|^2 / ||b_n||^2 over continuous angles (alternating 1D searches)."""
    R3 = R.reshape(model.N, model.G, model.nr)
    for sw in range(sweeps):
        W = np.einsum("nr,ngr->ng", model.rx_response(aoa).conj(), R3)

        def f_t(th):
            tp = model.tx_projection(np.asarray(th))
            return np.sum(np.abs(np.einsum("cng,ng->cn", tp.conj(), W)) ** 2 / np.sum(np.abs(tp) ** 2, axis=2), axis=1)
        aod = maximize_1d(f_t, aod, hw_t)
        tp0 = model.tx_projection(aod)
        V = np.einsum("ng,ngr->nr", tp0.conj(), R3)
        den = np.sum(np.abs(tp0) ** 2, axis=1)

        def f_r(th):
            ar = model.rx_response(np.asarray(th))
            return np.sum(np.abs(np.einsum("cnr,nr->cn", ar.conj(), V)) ** 2 / den, axis=1)
        aoa = maximize_1d(f_r, aoa, hw_r)
        hw_t, hw_r = hw_t / 4, hw_r / 4
    return aod, aoa


def _project(yb, omegas):
    """Per-subcarrier LS of yb onto the atoms (Eqs. 46-48 combined). Returns (coef [K, N], residual)."""
    if not omegas:
        return np.zeros((0, yb.shape[0]), dtype=complex), yb.copy()
    cols = np.stack(omegas, axis=2)  # [N, M, K]
    coef = np.stack([np.linalg.lstsq(cols[n], yb[n], rcond=None)[0] for n in range(yb.shape[0])], axis=1)
    return coef, yb - np.einsum("nmk,kn->nm", cols, coef)


def dcs_somp_offgrid(y, context, operator, sin_t, sin_r, *, threshold, max_atoms, cyclic_passes=2, min_atoms=0,
                     proposals=None, continue_search=False):
    """Algorithm 1 with continuous atoms. After every accepted atom, all atoms are re-refined cyclically
    on their leave-one-out residuals (as in Newtonized OMP), so that model mismatch of earlier atoms does
    not create spurious detections at high SNR."""
    cfg = context.config
    e = cfg.estimator
    model = ObservationModel(context)
    lam_c, d = cfg.physics.wavelength_c, cfg.antenna_spacing
    yb = beamspace.stack_observation(y).astype(complex)
    norms = operator.column_norms()
    R = yb.copy()
    omegas, captured, atoms, aods, aoas = [], [], [], [], []
    rejected = None
    res_t = lambda th: angle_resolution(th, cfg.array.num_tx, lam_c, d) / e.dictionary_oversampling
    res_r = lambda th: angle_resolution(th, cfg.array.num_rx, lam_c, d) / e.dictionary_oversampling
    # proposals: optional list of (aod, aoa) candidates from an external (e.g. learned) front end, tried in order
    # instead of the correlation argmax (42); acceptance still uses the threshold test (41) on the residual
    # continue_search: after the proposals are exhausted, continue with the correlation argmax (42)
    queue = list(proposals) if proposals is not None else None
    for _ in range(max_atoms + (len(queue) if queue is not None else 0)):
        if len(atoms) >= max_atoms:
            break
        if queue is not None and not queue:
            if not continue_search:
                break
            queue = None
        if queue is None:
            j = int(np.argmax(np.sum(np.abs(operator.correlate(R)) / norms, axis=0)))  # Eq. (42)
            it, ir = beamspace.split_index(j, operator.nbr)
            aod0 = float(beamspace.sine_to_angle(sin_t[it], e.aod_branch))
            aoa0 = float(beamspace.sine_to_angle(sin_r[ir], e.aoa_branch))
        else:
            j = -1
            aod0, aoa0 = queue.pop(0)
        aod, aoa = _refine_atom(model, R, aod0, aoa0, 1.2 * res_t(aod0), 1.2 * res_r(aoa0))
        if proposals is not None and any(abs(np.sin(aod) - np.sin(a)) < 2.0 / cfg.array.num_tx and
                                     abs(np.sin(aoa) - np.sin(b)) < 2.0 / cfg.array.num_rx for a, b in zip(aods, aoas)):
            if queue is None:
                break  # the argmax search itself returned to an accepted path: nothing new left
            continue  # proposal refined onto an already accepted path (within one bin): duplicate
        omega = _atom(model, aod, aoa)
        coef_new, R_new = _project(yb, omegas + [omega])
        energy = float(np.sum(np.abs(R) ** 2) - np.sum(np.abs(R_new) ** 2))  # sum_n ||r_{t-1} - r_t||^2
        if energy < threshold and len(atoms) >= min_atoms:
            rejected = energy
            if queue is None:
                break
            continue  # proposals: a rejected candidate does not end the search
        atoms.append(j); omegas.append(omega); captured.append(energy); aods.append(aod); aoas.append(aoa)
        for _ in range(cyclic_passes if len(omegas) > 1 else 0):
            for k in range(len(omegas)):
                coef, _ = _project(yb, omegas)
                others = [l for l in range(len(omegas)) if l != k]
                Rk = yb - sum(coef[l][:, None] * omegas[l] for l in others)
                aods[k], aoas[k] = _refine_atom(model, Rk, aods[k], aoas[k], 0.3 * res_t(aods[k]),
                                                0.3 * res_r(aoas[k]), sweeps=2)
                omegas[k] = _atom(model, aods[k], aoas[k])
        _, R = _project(yb, omegas)
    coef, R = _project(yb, omegas)
    res = DCSSOMPResult(np.array(atoms, dtype=int), coef, R, captured, threshold, rejected)
    return res, np.array(aods), np.array(aoas)


# --------------------------------------------------------------------------------------
def coarse_estimate(y: np.ndarray, context, proposals=None, continue_search: bool = False) -> ChannelEstimate:
    """y [N, G, Nr] -> ChannelEstimate in detection order (info holds the raw DCS-SOMP result).

    proposals: optional [(aod, aoa), ...] candidate atoms (off-grid detection only), see dcs_somp_offgrid."""
    cfg = context.config
    e = cfg.estimator
    Ut, Ur, sin_t, sin_r = beamspace.dictionaries_from_config(cfg)
    op = beamspace.SensingOperator(context.s, Ut, Ur)
    thr = None
    if e.num_paths is None:
        # grid DCS-SOMP: the max is over the dictionary columns (41); off-grid: the max is over continuous angles,
        # whose effective number of independent cells is calibrated relative to Nt*Nr whatever the oversampling (F23)
        if e.detection == "dcs_somp_offgrid":
            eff_cols = cfg.array.num_tx * cfg.array.num_rx * e.offgrid_threshold_factor
        else:
            eff_cols = op.num_columns
        thr = detection_threshold(context.n0, cfg.physics.num_subcarriers, eff_cols, e.pfa)
    if e.detection == "dcs_somp":
        yb = beamspace.stack_observation(y)
        res = dcs_somp(yb, op, num_atoms=e.num_paths, threshold=thr, max_atoms=e.max_paths, min_atoms=e.min_paths)
        it, ir = beamspace.split_index(res.atoms, op.nbr)
        aod = np.atleast_1d(beamspace.sine_to_angle(sin_t[it], e.aod_branch))
        aoa = np.atleast_1d(beamspace.sine_to_angle(sin_r[ir], e.aoa_branch))
        z = np.array([coupling_factor(Ut[:, it[k]], Ur[:, ir[k]], aod[k], aoa[k], cfg) for k in range(res.num_paths)])
    elif e.detection == "dcs_somp_offgrid":
        max_atoms = e.num_paths if e.num_paths is not None else e.max_paths
        res, aod, aoa = dcs_somp_offgrid(y, context, op, sin_t, sin_r,
                                         threshold=thr if thr is not None else -np.inf, max_atoms=max_atoms,
                                         min_atoms=e.min_paths, proposals=proposals,
                                         continue_search=continue_search)
        it, ir = beamspace.split_index(res.atoms, op.nbr)
        z = np.ones(res.num_paths)  # atoms are exact steering vectors: coefficient = h~ exp(-j2pi n tau/NTs)
    else:
        raise ValueError(f"unknown detection {e.detection!r}")
    Ts = cfg.physics.sampling_period
    K = res.num_paths
    delay = np.zeros(K)
    gain = np.zeros(K, dtype=complex)
    for k in range(K):
        coef = res.coefficients[k]
        if e.delay_method == "phase_slope":
            delay[k] = delay_estimation.phase_slope_delay(coef, Ts)
        elif e.delay_method == "periodogram":
            delay[k] = delay_estimation.periodogram_delay(coef, Ts, e.delay_oversampling)
        else:
            raise ValueError(f"unknown delay_method {e.delay_method!r}")
        gain[k] = delay_estimation.gain_from_delay(coef, delay[k], z[k], Ts)
    return ChannelEstimate(delay, np.asarray(aod, float), np.asarray(aoa, float), gain,
                           info={"dcs_somp": res, "tx_index": it, "rx_index": ir})

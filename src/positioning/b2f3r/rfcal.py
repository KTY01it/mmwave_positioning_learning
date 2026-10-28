"""T7: calibration of front-end impairments from pilots only (no ground truth, no reference device).

The impairment models are those of positioning.robust.rf (simulation stress tests, design-chosen magnitudes).

  CFO                 identifiable from ONE observation: the per-symbol phase advance of the dominant (delay, AoA) cell
                      is estimated jointly with its AoD (data-aided frequency estimate on a known sequence), optionally
                      refined with the phase slope of the fitted multipath model; the time-domain rotation is then
                      undone exactly.
  per-chain phase and gain (Rx and Tx), and the direction-independent part of element-position errors
                      Rx gains are identifiable from one observation; Tx gains are not (Nt = 65 unknowns, G * K beam
                      measurements), so the device is self-calibrated from L unlabelled pilot observations by
                      alternating least squares on a multipath model (paths from the uncalibrated estimator).
                      A linear phase gradient over an array is equivalent to a pointing offset and is NOT identifiable
                      without a reference direction.
  timing offset       equivalent to a clock bias on every delay: see `clock_bias_bound`; not corrected from pilots.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..robust import rf as RF
from ..signal_processing import beamspace
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..system import geometry
from ..system.arrays import ula_response

N_CP = RF.N_CP


def device(tag: str, index: int, nt: int = 65, nr: int = 65) -> dict:
    """Unit draws of a device of the B2f3-Robust program (same layout as robust.rf.device, different namespace, no
    repetition): used for confirmation devices that never occurred in development."""
    import hashlib
    ent = int.from_bytes(hashlib.sha256(f"icc2027/b2f3-robust/v1/device/{tag}".encode()).digest(), "big")
    rng = np.random.default_rng(np.random.SeedSequence(entropy=ent, spawn_key=(int(index),)))
    return {"id": int(index), "p_tx": rng.standard_normal(nt), "p_rx": rng.standard_normal(nr),
            "a_tx": rng.standard_normal(nt), "a_rx": rng.standard_normal(nr),
            "s_tx": rng.standard_normal((2, nt)), "s_rx": rng.standard_normal((2, nr))}


# ------------------------------------------------------------------------------------------------ impairment (multi)
def impair(sample, levels: dict, dev: dict):
    """`sample` (keep_noiseless=True) under several impairments at once: levels = {"P": deg, "A": dB, "F": ppm,
    "T": samples, "S": lambda}. Same unit draws, formulas and noise handling as robust.rf.impair."""
    ctx = sample.context
    cfg = ctx.config
    s = ctx.s
    N, G, _ = s.shape
    ct = np.exp(1j * np.deg2rad(levels.get("P", 0.0)) * dev["p_tx"]) * 10 ** (levels.get("A", 0.0) * dev["a_tx"] / 20)
    cr = np.exp(1j * np.deg2rad(levels.get("P", 0.0)) * dev["p_rx"]) * 10 ** (levels.get("A", 0.0) * dev["a_rx"] / 20)
    H = sample.h
    if levels.get("S", 0.0):
        p = sample.gt.paths
        k_n = 2 * np.pi / ctx.wavelengths
        sig = levels["S"] * cfg.physics.wavelength_c

        def pert(theta, u):
            proj = np.sin(theta)[:, None] * u[0][None, :] + np.cos(theta)[:, None] * u[1][None, :]
            return np.exp(-1j * sig * k_n[None, :, None] * proj[:, None, :])
        H = np.einsum("knrt,knr,knt->nrt", RF._path_channels(sample), pert(p.aoa, dev["s_rx"]), pert(p.aod, dev["s_tx"]).conj())
    clean = np.einsum("nrt,ngt->ngr", H, s * ct) * cr
    if levels.get("T", 0.0):
        clean = clean * np.exp(-2j * np.pi * np.arange(N) * levels["T"] / N)[:, None, None]
    if levels.get("F", 0.0):
        clean = rotate_cfo(clean, levels["F"] * 1e-6 * cfg.physics.fc * N / cfg.physics.bandwidth)
    return dataclasses.replace(sample, y=clean + (sample.y - sample.noiseless))


# ----------------------------------------------------------------------------------------------------------- CFO
def rotate_cfo(y, eps: float):
    """Time-domain rotation e^{j 2 pi eps t / N} over G sequential OFDM symbols of N + N_CP samples (eps in subcarrier
    spacings). rotate_cfo(y, -eps) is the exact inverse; the transform is unitary, so white noise stays white."""
    N, G = y.shape[:2]
    t = np.arange(G)[None, :] * (N + N_CP) + N_CP + np.arange(N)[:, None]
    return np.fft.fft(np.fft.ifft(y, axis=0) * np.exp(2j * np.pi * eps * t / N)[:, :, None], axis=0)


def _cells(y, context):
    cfg = context.config
    Ur = beamspace.dictionary(cfg.array.num_rx, cfg.physics.wavelength_c, cfg.antenna_spacing, cfg.array.index_origin,
                              1, "centered", True)
    yd = y * np.conj(context.x[:, :, 0])[:, :, None]
    return np.fft.ifft(yd @ np.conj(Ur), axis=0) * np.sqrt(y.shape[0])          # [delay, G, rx-beam]


def estimate_cfo(y, context, phi_max: float = 0.6, nfft: int = 512, os: int = 4) -> float:
    """CFO in subcarrier spacings from one observation. Dominant (delay, receive-beam) cell z_g, then the joint maximum
    over the AoD grid and the per-symbol phase advance phi of |sum_g z_g conj(b_g(theta)) e^{-j phi g}|^2 / ||b||^2."""
    cfg = context.config
    N, G = y.shape[:2]
    nt = cfg.array.num_tx
    c = _cells(y, context)
    d, b = np.unravel_index(int(np.argmax(np.sum(np.abs(c) ** 2, axis=1))), (c.shape[0], c.shape[2]))
    z = c[d, :, b]
    grid = np.arcsin(np.clip(np.linspace(-1, 1, os * nt, endpoint=False) + 1.0 / (os * nt), -1, 1))
    a = ula_response(grid, nt, np.array([cfg.physics.wavelength_c]), cfg.antenna_spacing, cfg.array.index_origin)[:, 0, :]
    B = a.conj() @ context.F[0, :, :, 0].T                                       # [C, G]
    S = np.abs(np.fft.fft(z[None, :] * B.conj(), nfft, axis=1)) ** 2 / np.maximum(np.sum(np.abs(B) ** 2, axis=1), 1e-30)[:, None]
    phi = 2 * np.pi * np.fft.fftfreq(nfft)
    S[:, np.abs(phi) > phi_max] = 0.0
    ci, pi = np.unravel_index(int(np.argmax(S)), S.shape)
    l, m, r = S[ci, (pi - 1) % nfft], S[ci, pi], S[ci, (pi + 1) % nfft]          # parabolic interpolation
    off = 0.5 * (l - r) / (l - 2 * m + r) if (l - 2 * m + r) != 0 else 0.0
    return float((phi[pi] + off * 2 * np.pi / nfft) * N / (2 * np.pi * (N + N_CP)))


def refine_cfo(y, context, est: ChannelEstimate) -> float:
    """Residual CFO (subcarrier spacings) from the weighted linear fit of the per-symbol phase of <model, y>."""
    N, G = y.shape[:2]
    mu = ObservationModel(context).mean(est)
    q = np.einsum("ngr,ngr->g", mu.conj(), y)
    w = np.abs(q)
    g = np.arange(G)
    psi = np.angle(q * np.conj(np.sum(q) / max(abs(np.sum(q)), 1e-300)))
    gm = np.sum(w * g) / np.sum(w)
    slope = np.sum(w * (g - gm) * (psi - np.sum(w * psi) / np.sum(w))) / max(np.sum(w * (g - gm) ** 2), 1e-300)
    return float(slope * N / (2 * np.pi * (N + N_CP)))


# ------------------------------------------------------------------------------------- per-chain gain calibration
def apply_gains(sample, c_tx, c_rx):
    """Compensated sample: y / c_rx (exact signal model at the receiver) and beams F * c_tx (exact transmit model).
    The receiver noise becomes N0 / |c_rx|^2 per element; context.n0 is set to its mean over the elements."""
    ctx = sample.context
    y = sample.y / c_rx[None, None, :]
    F = ctx.F * c_tx[None, None, :, None]
    n0 = float(ctx.n0 * np.mean(1.0 / np.abs(c_rx) ** 2))
    return dataclasses.replace(sample, y=y, context=dataclasses.replace(ctx, F=F, n0=n0))


def _terms(context, est: ChannelEstimate):
    """Per path: e [N], a_rx [N, Nr], a_tx [N, Nt] of the ideal array."""
    cfg = context.config
    m = ObservationModel(context)
    return [(m.delay_phasor(est.delay[k]), m.rx_response(est.aoa[k]),
             ula_response(est.aod[k], m.nt, m.lam, m.d, m.origin)) for k in range(est.num_paths)], m


def calibrate_gains(snaps, paths, n_iter: int = 4, ridge: float = 1e-3):
    """Alternating least squares for per-chain complex gains (c_tx [Nt], c_rx [Nr]) of one device.

    snaps: list of (y [N, G, Nr], context); paths: list of ChannelEstimate with the delays and angles held fixed.
    Model: y[n,g,r] = c_rx[r] sum_k g_k e_k[n] a_rx,k[n,r] sum_t conj(a_tx,k[n,t]) c_tx[t] s[n,g,t].
    Gauge: mean(c_tx) = mean(c_rx) = 1 (a common complex factor is absorbed by the path gains)."""
    nt = snaps[0][1].config.array.num_tx
    nr = snaps[0][1].config.array.num_rx
    ct, cr = np.ones(nt, complex), np.ones(nr, complex)
    T = [_terms(ctx, p) for (_, ctx), p in zip(snaps, paths)]
    for _ in range(n_iter):
        num, den = np.zeros(nr, complex), np.zeros(nr)
        A, bvec = np.zeros((nt, nt), complex), np.zeros(nt, complex)
        for (y, ctx), (terms, m) in zip(snaps, T):
            s = m.s                                                               # [N, G, Nt]
            Q = [np.conj(at)[:, None, :] * s for (_, _, at) in terms]             # [N, G, Nt] per path
            u = [q @ ct for q in Q]                                               # [N, G]
            basis = np.stack([(e[:, None] * uk)[:, :, None] * (ar * cr[None, :])[:, None, :] for (e, ar, _), uk in zip(terms, u)])
            Bm = basis.reshape(len(terms), -1)
            g = np.linalg.lstsq(Bm.T, y.ravel(), rcond=None)[0]
            mR = np.einsum("k,kngr->ngr", g, basis) / cr[None, None, :]
            num += np.einsum("ngr,ngr->r", mR.conj(), y)
            den += np.einsum("ngr,ngr->r", mR.conj(), mR).real
            P = [gk * e[:, None] * ar * cr[None, :] for gk, (e, ar, _) in zip(g, terms)]      # [N, Nr], includes c_rx
            for k in range(len(terms)):
                v = np.einsum("nr,ngr->ng", P[k].conj(), y)                        # [N, G]
                bvec += np.einsum("ngt,ng->t", Q[k].conj(), v)
                for j in range(len(terms)):
                    w = np.einsum("nr,nr->n", P[k].conj(), P[j])                   # [N]
                    A += np.einsum("ngt,n,ngu->tu", Q[k].conj(), w, Q[j], optimize=True)
        cr = num / np.maximum(den, 1e-300)
        lam = ridge * np.real(np.trace(A)) / nt
        ct = np.linalg.solve(A + lam * np.eye(nt), bvec + lam * np.ones(nt))
        ct, cr = ct / np.mean(ct), cr / np.mean(cr)
    return ct, cr


def self_calibrate(snaps, path_fn, rounds: int = 2, strong: float = 0.1):
    """Device self-calibration from unlabelled pilot observations.

    snaps: list of SimulationOutput-like objects (y, context); path_fn(sample) -> ChannelEstimate (the uncalibrated
    estimator's path stage). Round 1 uses the strongest path of every observation; later rounds re-estimate the paths
    on the compensated observations and keep those with |gain|^2 >= strong * max."""
    nt, nr = snaps[0].context.config.array.num_tx, snaps[0].context.config.array.num_rx
    ct, cr = np.ones(nt, complex), np.ones(nr, complex)
    for rnd in range(rounds):
        cur = [apply_gains(s, ct, cr) for s in snaps] if rnd else snaps
        paths = []
        for s in cur:
            e = path_fn(s)
            pw = np.abs(e.gain_tilde) ** 2
            keep = [int(np.argmax(pw))] if rnd == 0 else [k for k in range(e.num_paths) if pw[k] >= strong * pw.max()]
            paths.append(e.subset(keep))
        ct, cr = calibrate_gains([(s.y, s.context) for s in snaps], paths)
    return ct, cr


# ----------------------------------------------------------------------------------------------- timing / clock bias
def clock_bias_bound(sample):
    """(PEB, PEB with an unknown common delay bias, rank deficiency) at the true parameters: is a timing offset
    identifiable from this observation? Uses the frozen channel FIM and geometry Jacobian."""
    from ..bounds.fim import channel_fim
    g = sample.gt
    c = sample.context.config.physics.speed_of_light
    has_los = bool(np.any(g.paths.is_los))
    J = channel_fim(sample.context, g.paths)
    T = geometry.transformation_matrix(g.bs_position, g.position, g.orientation, g.scatterers, has_los, c)
    K = g.paths.num_paths
    tb = np.zeros((1, 5 * K))
    tb[0, 0::5] = 1.0
    Ta = np.vstack([T, tb])
    out = []
    for M in (T, Ta):
        Jx = M @ J @ M.T
        sv = np.linalg.svd(Jx / np.outer(np.sqrt(np.diag(Jx)), np.sqrt(np.diag(Jx))), compute_uv=False)
        if sv[-1] < 1e-10 * sv[0]:
            out.append((float("inf"), 1))
        else:
            C = np.linalg.inv(Jx)
            out.append((float(np.sqrt(max(C[0, 0] + C[1, 1], 0.0))), 0))
    return out[0][0], out[1][0], out[1][1]

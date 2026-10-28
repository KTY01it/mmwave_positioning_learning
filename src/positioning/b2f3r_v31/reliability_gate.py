"""R2: cheap online reliability statistics of the v2 output and gates for the v3 hypothesis stage.

All features use only y, the configuration, the v2 path-stage estimate H0 (its residual energy f0 and residual-floor noise
level n0_hat), the support v2 used and its localisation; nothing that needs ground truth, the simulator N0 or the condition
label. Cost: one residual, one angle-delay correlation (the frozen `_residual_peak` computation) and K least-squares gain
fits (~ms)."""
from __future__ import annotations

import numpy as np

from ..b2f3r import noise as NZ1
from ..models.features import correlation_cube
from ..models.path_init import ls_gains
from ..models.v3_aosage import _fit
from ..signal_processing.model import ObservationModel

Z = 66.71


def features(y, ctx0, e0, f0, n0, out, log, base_est=None, base_f=None) -> dict:
    """ctx0: context with n0 = n0_hat of H0. Returns F1..F8 (floats)."""
    cfg = ctx0.config
    model = ObservationModel(ctx0)
    M = y.size
    K = e0.num_paths
    r = y - model.mean(e0)
    c = correlation_cube(r, ctx0)                                   # already normalised by sqrt(n0_hat)
    P = np.abs(np.fft.ifft(c, axis=0) * np.sqrt(c.shape[0])) ** 2
    rm = []                                                         # NLL increase when one path is removed (LS gains)
    if K >= 2:
        for k in range(K):
            s = e0.subset([j for j in range(K) if j != k])
            rm.append((_fit(model, y, type(e0)(s.delay, s.aod, s.aoa, ls_gains(model, y, s.delay, s.aod, s.aoa))) - f0) / n0)
    snr = [float(np.sum(np.abs(model.path_signal(e0.delay[k], e0.aod[k], e0.aoa[k], e0.gain_tilde[k])) ** 2) / n0) for k in range(K)]
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    sep = np.inf
    for a in range(K):
        for b in range(a + 1, K):
            sep = min(sep, max(abs(e0.delay[a] - e0.delay[b]) / Ts, abs(np.sin(e0.aod[a]) - np.sin(e0.aod[b])) * nt / 2,
                               abs(np.sin(e0.aoa[a]) - np.sin(e0.aoa[b])) * nr / 2))
    cost = getattr(out, "cost", np.nan)
    return {
        "F1_stop_margin": float(f0 / n0 / NZ1.ratio_threshold(M) - 1.0),    # < 0: residual consistent with noise
        "F2_peak": float(P.max()),                                         # strongest residual cell / n0_hat
        "F2_peak_rel": float(P.max() / max(P.mean(), 1e-30)),
        "F3_fwd_entered": float(log.get("fwd_entered", 0) > 0), "F3_removed": float(log.get("removed", 0)),
        "F3_accepted": float(log.get("accepted", 0)), "F3_switched": float(log.get("switched", 0)),
        "F5_min_removal": float(min(rm)) if rm else np.inf,               # weakest path's NLL contribution
        "F5_min_snr": float(min(snr)) if snr else np.nan,
        "F6_min_sep": float(sep),
        "F7_cost": float(cost) if cost is not None and np.isfinite(cost) else np.inf,
        "F7_cls_start": float(out.info.get("geo") == "cls1"),
        "F8_K": float(K), "F8_repair": float(log.get("support_repair", 0)), "F8_G": float(y.shape[1]),
    }


def deterministic(fe, tau_peak, tau_rm):
    """G1: run Hp only if the strongest residual cell exceeds tau_peak; run Hm only if the weakest path contributes less
    than tau_rm. Returns (run_Hp, run_Hm)."""
    return fe["F2_peak"] > tau_peak, fe["F5_min_removal"] < tau_rm

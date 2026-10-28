"""Offline diagnostics (ground truth used for scoring only, never at inference)."""
from __future__ import annotations

import numpy as np

from ..robust.evalkit import match
from ..signal_processing.model import ChannelEstimate, ObservationModel


def path_report(est, sample) -> dict:
    """Per true path: SNR of its signal (energy / N0, dB), matched flag and errors of the matched estimate."""
    cfg = sample.context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    gp = sample.gt.paths
    model = ObservationModel(sample.context)
    snr = [float(10 * np.log10(np.sum(np.abs(model.path_signal(gp.delay[k], gp.aod[k], gp.aoa[k], gp.gain_tilde[k])) ** 2)
                               / sample.context.n0)) for k in range(gp.num_paths)]
    out = {"snr_db": snr, "matched": [0] * gp.num_paths, "d_tau_ns": [None] * gp.num_paths,
           "d_sin_aod": [None] * gp.num_paths, "d_sin_aoa": [None] * gp.num_paths}
    if est is None or est.num_paths == 0:
        return out
    m = match((np.asarray(gp.delay), np.asarray(gp.aod), np.asarray(gp.aoa)), (est.delay, est.aod, est.aoa), Ts, nt, nr)
    for i, j in m.items():
        out["matched"][i] = 1
        out["d_tau_ns"][i] = float((est.delay[j] - gp.delay[i]) * 1e9)
        out["d_sin_aod"][i] = float(np.sin(est.aod[j]) - np.sin(gp.aod[i]))
        out["d_sin_aoa"][i] = float(np.sin(est.aoa[j]) - np.sin(gp.aoa[i]))
    out["support_ok"] = int(len(m) == gp.num_paths and est.num_paths == gp.num_paths)
    out["khat"] = int(est.num_paths)
    return out

"""Offline diagnostics of B2f3R-v3.1 (R0 cost decomposition, R1 failure taxonomy). GROUND TRUTH IS USED HERE ONLY FOR
SCORING AND ORACLE SUBSTITUTIONS; nothing in this module is imported by the estimator."""
from __future__ import annotations

import numpy as np

from ..signal_processing.model import ChannelEstimate


def true_estimate(sample) -> ChannelEstimate:
    gp = sample.gt.paths
    return ChannelEstimate(np.asarray(gp.delay, float), np.asarray(gp.aod, float), np.asarray(gp.aoa, float), np.asarray(gp.gain_tilde))


def path_taxonomy(est, sample) -> dict:
    """Per-path evidence for M1-M5 (operational definitions, docs/b2f3r_v31/r1_failure_taxonomy.md).

    A true path k is 'found' if an estimated path lies within one delay sample, 2/Nt in sin AoD and 2/Nr in sin AoA
    (greedy, strongest true path first). For an unfound true path, the nearest unused estimated path in delay
    (|dtau| <= 1.5 Ts) is inspected: AoD-lobe error if |dsin AoD| > 2/Nt and |dsin AoA| <= 2/Nr; AoA-lobe error in the
    mirrored case; branch ambiguity if |dsin| <= 2/N but |dangle| > 0.2 rad (front/back symmetry, exact for a ULA)."""
    cfg = sample.context.config
    Ts, nt, nr = cfg.physics.sampling_period, cfg.array.num_tx, cfg.array.num_rx
    t = true_estimate(sample)
    from ..signal_processing.model import ObservationModel
    model = ObservationModel(sample.context)
    snr = [float(10 * np.log10(np.sum(np.abs(model.path_signal(t.delay[k], t.aod[k], t.aoa[k], t.gain_tilde[k])) ** 2) / sample.context.n0))
           for k in range(t.num_paths)]
    order = np.argsort(snr)[::-1]
    K = 0 if est is None else est.num_paths
    used, found, lobe = set(), {}, {}
    for k in order:
        for j in range(K):
            if j in used:
                continue
            if (abs(est.delay[j] - t.delay[k]) <= Ts and abs(np.sin(est.aod[j]) - np.sin(t.aod[k])) <= 2 / nt
                    and abs(np.sin(est.aoa[j]) - np.sin(t.aoa[k])) <= 2 / nr):
                used.add(j); found[int(k)] = j
                break
    for k in order:
        if int(k) in found:
            continue
        cand = [j for j in range(K) if j not in used and abs(est.delay[j] - t.delay[k]) <= 1.5 * Ts]
        if not cand:
            continue
        j = min(cand, key=lambda j: abs(est.delay[j] - t.delay[k]))
        dsd, dsa = abs(np.sin(est.aod[j]) - np.sin(t.aod[k])), abs(np.sin(est.aoa[j]) - np.sin(t.aoa[k]))
        kind = []
        if dsd > 2 / nt:
            kind.append("aod_lobe")
        if dsa > 2 / nr:
            kind.append("aoa_lobe")
        if dsa <= 2 / nr and abs(est.aoa[j] - t.aoa[k]) > 0.2:
            kind.append("aoa_branch")
        if dsd <= 2 / nt and abs(est.aod[j] - t.aod[k]) > 0.2:
            kind.append("aod_branch")
        lobe[int(k)] = {"j": j, "kind": kind, "dsin_aod": float(dsd), "dsin_aoa": float(dsa)}
        used.add(j)
    merged = 0
    for a in range(t.num_paths):                      # two true paths within one resolution cell of each other
        for b in range(a + 1, t.num_paths):
            if (abs(t.delay[a] - t.delay[b]) <= Ts and abs(np.sin(t.aod[a]) - np.sin(t.aod[b])) <= 2 / nt
                    and abs(np.sin(t.aoa[a]) - np.sin(t.aoa[b])) <= 2 / nr and (int(a) not in found or int(b) not in found)):
                merged += 1
    errs = {int(k): {"dtau_ns": float((est.delay[j] - t.delay[k]) * 1e9), "dsin_aod": float(np.sin(est.aod[j]) - np.sin(t.aod[k])),
                     "dsin_aoa": float(np.sin(est.aoa[j]) - np.sin(t.aoa[k]))} for k, j in found.items()}
    missing = [int(k) for k in range(t.num_paths) if int(k) not in found and int(k) not in lobe]
    return {"K_true": int(t.num_paths), "K_est": int(K), "snr_true_db": [round(v, 1) for v in snr], "found": {str(k): j for k, j in found.items()},
            "missing": missing, "missing_snr_db": [round(snr[k], 1) for k in missing], "lobe": {str(k): v for k, v in lobe.items()},
            "spurious": int(K - len(used)), "merged_pairs": merged, "errors": {str(k): v for k, v in errs.items()},
            "support_ok": int(len(found) == t.num_paths and K == t.num_paths)}


def substituted(est, sample, tax) -> ChannelEstimate | None:
    """ORACLE: the estimated support with every found path's parameters replaced by the true ones (M6 test)."""
    if est is None or not tax["found"]:
        return None
    t = true_estimate(sample)
    d, a, b, g = (np.array(v, dtype=complex if i == 3 else float) for i, v in enumerate((est.delay, est.aod, est.aoa, est.gain_tilde)))
    for k, j in tax["found"].items():
        k = int(k)
        d[j], a[j], b[j], g[j] = t.delay[k], t.aod[k], t.aoa[k], t.gain_tilde[k]
    return ChannelEstimate(d, a, b, g)

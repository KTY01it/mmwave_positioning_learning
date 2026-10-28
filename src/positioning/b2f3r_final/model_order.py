"""F2 candidate model-order interventions [DESIGN/CANDIDATE]. Each produces one competing support for the frozen B4 selection.

M1 relative-significance pruning: remove every path of the base support whose estimated SNR (path signal energy / n0_hat)
   is more than `rho_db` below that of the strongest path, then refit with the frozen SAGE step. Unlike pruning to a fixed
   number of paths it imposes no model order; it targets the high-SNR failure in which weak spurious paths pass the GLRT
   because the residual is dominated by model mismatch rather than noise (R1 of v3.1: M5 at 15-25 dB)."""
from __future__ import annotations

import numpy as np

from ..b2f3r_v3.hypotheses import _refit
from ..models.path_init import ls_gains
from ..models.v3_aosage import _fit
from ..signal_processing.model import ChannelEstimate, ObservationModel


def path_snr_db(est, context):
    model = ObservationModel(context)
    return np.array([10 * np.log10(max(np.sum(np.abs(model.path_signal(est.delay[k], est.aod[k], est.aoa[k], est.gain_tilde[k])) ** 2) / context.n0, 1e-30))
                     for k in range(est.num_paths)])


def relprune(y, context, est, rho_db, k_min=2):
    s = path_snr_db(est, context)
    keep = [int(k) for k in np.argsort(-s) if s[k] >= s.max() - rho_db]
    if len(keep) == est.num_paths or len(keep) < k_min:
        return None
    keep = sorted(keep)
    model = ObservationModel(context)
    sub = est.subset(keep)
    h = _refit(y, context, ChannelEstimate(sub.delay, sub.aod, sub.aoa, ls_gains(model, y, sub.delay, sub.aod, sub.aoa)))
    return h, _fit(model, y, h)


def prune_to_k(y, context, est, k):
    """DIAGNOSTIC helper (used with the TRUE path count only in offline oracle analyses)."""
    from ..b2f3r_v31.targeted_recovery import prune_to
    return prune_to(y, context, est, k)

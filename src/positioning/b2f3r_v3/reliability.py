"""V3-C3: local geometric conditioning of a fitted solution (EXIP framework): with the FIM-based weight W used by the
frozen EXIP fit and the geometry Jacobian T at the ESTIMATED geometry, J = T W T^T and the position standard deviation
proxy sqrt(trace(J^-1)[pos]). A diagnostic indicator only; it is not a bound at the truth and not a selector. [DESIGN]"""
from __future__ import annotations

import numpy as np

from ..localization.exip_lma import weight_matrix
from ..system import geometry


def position_std(out, context) -> float:
    e = out.channel
    if e is None or out.condition is None or not np.all(np.isfinite(out.position)):
        return float("nan")
    try:
        has_los = out.condition in ("los", "nlos")
        if out.condition == "los":
            return float("nan")
        scat = np.asarray(out.scatterers, float).reshape(-1, 2) if out.scatterers is not None else np.zeros((0, 2))
        K = int(has_los) + scat.shape[0]
        if K != e.num_paths:
            return float("nan")
        W = weight_matrix(e, context, context.config.estimator.exip_weighting)
        T = geometry.transformation_matrix(context.bs_position, out.position, out.orientation, scat, has_los,
                                           context.config.physics.speed_of_light)
        J = T @ W @ T.T
        C = np.linalg.pinv(J)
        return float(np.sqrt(max(C[0, 0] + C[1, 1], 0.0)))
    except Exception:
        return float("nan")

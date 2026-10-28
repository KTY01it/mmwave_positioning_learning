"""Level 1 report for the geometry (Eqs. 17-22, 26, 61-62, 65). Writes outputs/validation/geometry.json."""
import json

import _bootstrap  # noqa: F401
import numpy as np

from _bootstrap import ROOT
from positioning.config import load_config
from positioning.system import geometry as G
from positioning.system.scenario import sample_scenario

C = 299_792_000.0


def jacobian_fd_error(bs, p, alpha, scat, has_los):
    T = G.transformation_matrix(bs, p, alpha, scat, has_los, C)
    layout = G.geometry_parameter_layout(int(has_los) + len(scat), has_los)
    worst = 0.0
    for i, name in enumerate(layout):
        if name[0] == "h":
            continue
        eps = 1e-7
        dp, da, ds = np.zeros(2), 0.0, np.zeros_like(scat)
        if name == "px": dp[0] = eps
        elif name == "py": dp[1] = eps
        elif name == "alpha": da = eps
        else: ds[int(name[1:-1]) - int(has_los), 0 if name[-1] == "x" else 1] = eps
        f = lambda sgn: G.compute_path_geometry(bs, p + sgn * dp, alpha + sgn * da, scat + sgn * ds, has_los, C)
        a, b = f(1), f(-1)
        fd = np.stack([(a.delay - b.delay) * C, G.wrap_angle(a.aod - b.aod), G.wrap_angle(a.aoa - b.aoa)], 1) / (2 * eps)
        an = T[i].reshape(-1, 5)[:, :3] * [C, 1, 1]
        worst = max(worst, float(np.max(np.abs(an - fd))))
    return worst


def main():
    report = {}
    for name in ("los", "nlos", "olos"):
        cfg = load_config(ROOT / "configs" / "experiments" / f"{name}.yaml")
        sc = sample_scenario(cfg, np.random.default_rng(0))
        g = G.compute_path_geometry(sc.bs_position, sc.ue_position, sc.orientation, sc.scatterers, sc.include_los, C)
        entry = {"ue": sc.ue_position.tolist(), "alpha": sc.orientation, "scatterers": sc.scatterers.tolist(),
                 "delay_ns": (g.delay * 1e9).tolist(), "aod_rad": g.aod.tolist(), "aoa_rad": g.aoa.tolist(),
                 "path_length_m": g.distance.tolist(),
                 "jacobian_max_abs_err_vs_fd": jacobian_fd_error(sc.bs_position, sc.ue_position, sc.orientation,
                                                                 sc.scatterers, sc.include_los)}
        if sc.include_los:
            ph, ah = G.los_position_from_parameters(g.delay[0], g.aod[0], g.aoa[0], sc.bs_position, C)
            entry["los_inverse_err_m"] = float(np.linalg.norm(ph - sc.ue_position))
            entry["los_inverse_err_rad"] = abs(ah - sc.orientation)
        for k, s in enumerate(sc.scatterers):
            ref = G.paper_geometry_arccos(sc.bs_position, sc.ue_position, sc.orientation, s, C)
            kk = k + int(sc.include_los)
            entry.setdefault("arccos_form_max_diff", []).append(float(max(
                abs(ref["tauk"] - g.delay[kk]) * C, abs(G.wrap_angle(ref["aodk"] - g.aod[kk])),
                abs(G.wrap_angle(ref["aoak"] - g.aoa[kk])))))
        report[name] = entry
        print(name, json.dumps(entry))
    out = ROOT / "outputs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

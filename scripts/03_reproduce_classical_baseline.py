"""Level 3: paper-level reproduction of Shahmansoori et al. (E0 / E0b of the master plan).

Experiments (each writes outputs/level3/<name>.json and figures):
  los      Figs. 3-5  RMSE of tau0, theta_Tx0, theta_Rx0, p, alpha vs SNR with CRB/PEB/REB; SAGE iterations
  nlos     Figs. 6-8  same with LOS + scatterer s = [1.5, 0.4]
  nlos_paper          ablation of F21/F24: LOS = min_delay (paper) / min_cost / strongest, without path pruning
  olos     Fig. 9     OLOS, three scatterers, delta_alpha in {0.01, 0.05}
  unknown  Table I    cost ratio v_nlos / v_olos and decision accuracy (rules min_cost, chi2)
  beams    Fig. 2     PEB and error at CDF 0.9 vs G, constant total transmit power
  cdf      Fig. 10    CDF of the error: LOS, NLOS with 1 and 2 scatterers, SNR = -5 dB, G = 20
  g0       P1.6       Go/No-Go: DCS-SOMP init vs oracle init (LOS)

    python scripts/03_reproduce_classical_baseline.py los nlos --mc 300
    python scripts/03_reproduce_classical_baseline.py all --mc 100
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import _bootstrap  # noqa: F401
import numpy as np

from _bootstrap import ROOT
from positioning.bounds.peb_reb import compute_bounds
from positioning.config import load_config
from positioning.evaluation import metrics as M
from positioning.evaluation import plots
from positioning.localization.pipeline import ClassicalEstimator, OracleInitEstimator
from positioning.signal_processing.coarse import coarse_estimate
from positioning.system.simulator import MmWavePositioningSystem

EXP = ROOT / "configs" / "experiments"
OUT_ROOT = ROOT / "outputs" / "level3"
OUT = OUT_ROOT / "strong_classical"  # set from --estimator in main()
SNRS = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0]


# ======================================================================================
# per-sample workers (top level for multiprocessing)
# ======================================================================================
def _channel_errors(est, gt, match="delay"):
    """Errors of the estimated channel parameters per TRUE path (NaN if the path was not detected)."""
    K = gt.paths.num_paths
    err = {k: np.full(K, np.nan) for k in ("delay", "aod", "aoa")}
    if est is None or est.num_paths == 0:
        return err
    if match == "delay" and K == 1:
        r, c = np.array([int(np.argmin(est.delay))]), np.array([0])
    else:
        r, c = M.match_paths_by_angle(est.aod, est.aoa, gt.paths.aod, gt.paths.aoa)
    err["delay"][c] = est.delay[r] - gt.paths.delay[c]
    err["aod"][c] = M.wrap_angle(est.aod[r] - gt.paths.aod[c])
    err["aoa"][c] = M.wrap_angle(est.aoa[r] - gt.paths.aoa[c])
    return err


def run_sample(args, overrides=None, condition="auto", history_snrs=(), oracle=False, profile=None):
    cfg_path, seed, snr = args
    cfg = load_config(cfg_path, estimator=profile, **(overrides or {}))
    s = MmWavePositioningSystem(cfg).simulate(seed=int(seed), snr_db=float(snr))
    b = compute_bounds(s)
    est = (OracleInitEstimator(condition, seed=int(seed)) if oracle else
           ClassicalEstimator(condition, record_history=snr in history_snrs))
    t0 = time.perf_counter()
    out = est(s)
    runtime = time.perf_counter() - t0
    ch = out.channel if out.channel is not None else getattr(out.info.get("sage"), "estimate", None)
    ce = _channel_errors(ch, s.gt)
    coarse = out.info.get("coarse")
    cc = _channel_errors(coarse, s.gt) if coarse is not None else None
    rec = {
        "seed": int(seed), "snr": float(snr), "K_true": s.gt.num_paths,
        "K_hat": 0 if ch is None else ch.num_paths,
        "pos_err": float(M.position_error(out.position, s.gt.position)),
        "ori_err": float(M.angle_error(out.orientation, s.gt.orientation)) if np.isfinite(out.orientation) else np.nan,
        "peb": b.peb, "reb": b.reb,
        "crb_delay": b.crb_delay.tolist(), "crb_aod": b.crb_aod.tolist(), "crb_aoa": b.crb_aoa.tolist(),
        "err_delay": ce["delay"].tolist(), "err_aod": ce["aod"].tolist(), "err_aoa": ce["aoa"].tolist(),
        "runtime": runtime, "condition_hat": out.condition,
        "sage_iterations": int(out.info["sage"].iterations) if "sage" in out.info else 0,
        "sage_converged": bool(out.info["sage"].converged) if "sage" in out.info else False,
        # residual energy ||y - mu||^2 / N0 before (coarse) and after SAGE; noise only would give N*G*Nr
        "res_coarse": float(out.info["sage"].residual_energy[0] / s.meta.n0) if "sage" in out.info else np.nan,
        "res_sage": float(out.info["sage"].residual_energy[-1] / s.meta.n0) if "sage" in out.info else np.nan,
        "num_obs": int(s.y.size),
        "K_coarse": int(coarse.num_paths) if coarse is not None else 0,
    }
    if cc is not None:
        rec.update({"coarse_err_delay": cc["delay"].tolist(), "coarse_err_aod": cc["aod"].tolist(),
                    "coarse_err_aoa": cc["aoa"].tolist()})
    for key in ("delta_v", "v_nlos", "v_olos", "p_nlos"):
        if key in out.info:
            v = out.info[key]
            rec[key] = None if v is None else float(v)
    if "sage" in out.info and out.info["sage"].history:
        hist = []
        for h in out.info["sage"].history:
            e = _channel_errors(h, s.gt)
            hist.append([e["delay"].tolist(), e["aod"].tolist(), e["aoa"].tolist()])
        rec["history"] = hist
    return rec


def pmap(fn, items, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, items, chunksize=1))


# ======================================================================================
# aggregation helpers
# ======================================================================================
def _rms(x):
    return M.rmse(np.asarray(x, float))


def aggregate(recs, snrs, num_paths):
    out = {"snr": snrs, "paths": []}
    by = {s: [r for r in recs if r["snr"] == s] for s in snrs}
    for k in range(num_paths):
        d = {}
        for q in ("delay", "aod", "aoa"):
            d[f"rmse_{q}"] = [_rms([r[f"err_{q}"][k] for r in by[s]]) for s in snrs]
            d[f"crb_{q}"] = [float(np.sqrt(np.mean([r[f"crb_{q}"][k] ** 2 for r in by[s]]))) for s in snrs]
            d[f"detected_{q}"] = [float(np.mean([np.isfinite(r[f"err_{q}"][k]) for r in by[s]])) for s in snrs]
            if "coarse_err_delay" in recs[0]:
                d[f"coarse_rmse_{q}"] = [_rms([r[f"coarse_err_{q}"][k] for r in by[s]]) for s in snrs]
        out["paths"].append(d)
    out["rmse_pos"] = [_rms([r["pos_err"] for r in by[s]]) for s in snrs]
    out["peb"] = [float(np.sqrt(np.mean([r["peb"] ** 2 for r in by[s]]))) for s in snrs]
    out["rmse_ori"] = [_rms([r["ori_err"] for r in by[s]]) for s in snrs]
    out["reb"] = [float(np.sqrt(np.mean([r["reb"] ** 2 for r in by[s]]))) for s in snrs]
    out["failure_rate"] = [float(np.mean([not np.isfinite(r["pos_err"]) for r in by[s]])) for s in snrs]
    out["delta_pos_db"] = [M.delta_db(a, b) for a, b in zip(out["rmse_pos"], out["peb"])]
    out["delta_ori_db"] = [M.delta_db(a, b) for a, b in zip(out["rmse_ori"], out["reb"])]
    out["snr_threshold_pos"] = M.snr_threshold(snrs, out["delta_pos_db"], failure_rates=out["failure_rate"])
    out["snr_threshold_ori"] = M.snr_threshold(snrs, out["delta_ori_db"], failure_rates=out["failure_rate"])
    out["outlier_rate_3peb"] = [M.outlier_rate([r["pos_err"] for r in by[s]], [r["peb"] for r in by[s]]) for s in snrs]
    out["K_hat_equals_K"] = [float(np.mean([r["K_hat"] == r["K_true"] for r in by[s]])) for s in snrs]
    out["sage_iterations_mean"] = [float(np.mean([r["sage_iterations"] for r in by[s]])) for s in snrs]
    out["sage_converged"] = [float(np.mean([r["sage_converged"] for r in by[s]])) for s in snrs]
    # log-likelihood gain of SAGE (Eq. 11): (||y-mu_coarse||^2 - ||y-mu_sage||^2)/N0, and final residual / (N G Nr)
    out["sage_loglik_gain"] = [float(np.mean([r["res_coarse"] - r["res_sage"] for r in by[s]])) for s in snrs]
    out["residual_over_noise"] = [float(np.mean([r["res_sage"] / r["num_obs"] for r in by[s]])) for s in snrs]
    out["K_hat_mean"] = [float(np.mean([r["K_hat"] for r in by[s]])) for s in snrs]
    out["K_hat_hist"] = [np.bincount([r["K_hat"] for r in by[s]], minlength=9).tolist() for s in snrs]
    out["runtime_mean_s"] = [float(np.mean([r["runtime"] for r in by[s]])) for s in snrs]
    out["n"] = [len(by[s]) for s in snrs]
    return out


def history_rmse(recs, snr, k):
    hs = [r["history"] for r in recs if r["snr"] == snr and "history" in r]
    if not hs:
        return None
    L = min(len(h) for h in hs)
    return {q: [_rms([h[i][j][k] for h in hs]) for i in range(L)] for j, q in enumerate(("delay", "aod", "aoa"))}


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2, default=float))
    print(f"[{name}] wrote {OUT / (name + '.json')}")


def print_table(name, agg, k_list):
    print(f"\n[{name}] SNR   " + "  ".join(f"{s:>8.0f}" for s in agg["snr"]))
    for label, key in (("RMSE p", "rmse_pos"), ("PEB", "peb"), ("Delta_p dB", "delta_pos_db"),
                       ("RMSE alpha", "rmse_ori"), ("REB", "reb"), ("Delta_a dB", "delta_ori_db"),
                       ("P(K^=K)", "K_hat_equals_K"), ("P_out", "outlier_rate_3peb"), ("fail", "failure_rate"),
                       ("K^ mean", "K_hat_mean"), ("SAGE it", "sage_iterations_mean"), ("SAGE conv", "sage_converged"),
                       ("dLL SAGE", "sage_loglik_gain"), ("res/noise", "residual_over_noise"),
                       ("time s", "runtime_mean_s")):
        print(f"  {label:11s} " + "  ".join(f"{v:8.2e}" if abs(v) < 1e-2 or abs(v) > 1e3 else f"{v:8.3f}"
                                           for v in agg[key]))
    for k in k_list:
        p = agg["paths"][k]
        for q in ("delay", "aod", "aoa"):
            print(f"  path{k} {q:5s} RMSE/CRB dB " + "  ".join(
                f"{10 * np.log10(a ** 2 / b ** 2):8.2f}" if np.isfinite(a) else "     nan" for a, b in
                zip(p[f"rmse_{q}"], p[f"crb_{q}"])))
    print(f"  threshold SNR (Delta_p <= 1 dB): {agg['snr_threshold_pos']}, (Delta_alpha): {agg['snr_threshold_ori']}")


def rmse_figure(name, agg, k_list, title):
    panels = []
    for k in k_list:
        p = agg["paths"][k]
        panels += [dict(rmse=np.array(p["rmse_delay"]) * 1e9, bound=np.array(p["crb_delay"]) * 1e9, label=f"RMSE(tau{k})",
                        bound_label=f"sqrt CRB(tau{k})", ylabel="[ns]"),
                   dict(rmse=p["rmse_aoa"], bound=p["crb_aoa"], label=f"RMSE(theta_Rx,{k})",
                        bound_label=f"sqrt CRB(theta_Rx,{k})", ylabel="[rad]"),
                   dict(rmse=p["rmse_aod"], bound=p["crb_aod"], label=f"RMSE(theta_Tx,{k})",
                        bound_label=f"sqrt CRB(theta_Tx,{k})", ylabel="[rad]")]
    plots.figure_rmse_panels(agg["snr"], panels, title + " — channel parameters", OUT / f"{name}_channel.png")
    plots.figure_rmse_panels(agg["snr"], [
        dict(rmse=agg["rmse_ori"], bound=agg["reb"], label="RMSE(alpha)", bound_label="REB", ylabel="[rad]"),
        dict(rmse=agg["rmse_pos"], bound=agg["peb"], label="RMSE(p)", bound_label="PEB", ylabel="[m]")],
        title + " — position / orientation", OUT / f"{name}_pose.png")


# ======================================================================================
# experiments
# ======================================================================================
def _known_k(args, overrides, K):
    """--known-k: diagnostic, DCS-SOMP told the true number of paths (isolates F17)."""
    return {**(overrides or {}), "estimator.num_paths": K} if args.known_k else (overrides or {})


def exp_fixed(name, cfg_file, args, num_paths, condition="auto", overrides=None, history=(-20.0, -10.0, 0.0)):
    cfg_path = str(EXP / cfg_file)
    overrides = _known_k(args, overrides, num_paths)
    items = [(cfg_path, 1_000_000 + i, s) for s in SNRS for i in range(args.mc)]
    recs = pmap(partial(run_sample, overrides=overrides, condition=condition, history_snrs=history,
                        profile=args.estimator), items, args.workers)
    agg = aggregate(recs, SNRS, num_paths)
    agg["sage_history"] = {str(s): [history_rmse(recs, s, k) for k in range(num_paths)] for s in history}
    print_table(name, agg, range(num_paths))
    save(name, agg)
    rmse_figure(name, agg, range(num_paths), name.upper())
    return agg


def exp_olos(args):
    res = {}
    for step in (0.05, 0.01):
        key = f"olos_dalpha_{step}"
        ov = {"estimator.olos_alpha_step": step}
        if args.olos_top_k:
            ov["estimator.olos_lma_top_k"] = args.olos_top_k
        res[key] = exp_fixed(key, "olos.yaml", args, 3, overrides=ov, history=())
    return res


def exp_unknown(args):
    cfg_path = str(EXP / "olos.yaml")
    snrs = [-20.0, -10.0, 0.0, 10.0]
    ov = {"estimator.olos_alpha_step": 0.05}
    if args.olos_top_k:
        ov["estimator.olos_lma_top_k"] = args.olos_top_k
    n = args.unknown_mc
    out = {"snr": snrs}
    for data_name, data_cfg, data_ov in (("olos_data", cfg_path, {}),
                                         ("nlos_data", str(EXP / "nlos.yaml"),
                                          {"scenario.scatterers": {"mode": "fixed", "positions": [[1.5, 0.4], [1.5, 0.9]]}})):
        items = [(data_cfg, 2_000_000 + i, s) for s in snrs for i in range(n)]
        recs = pmap(partial(run_sample, overrides=_known_k(args, {**ov, **data_ov}, 3), condition="unknown", history_snrs=(),
                            profile=args.estimator), items, args.workers)
        truth = "olos" if data_name == "olos_data" else "nlos"
        d = {"mean_delta_v": [], "median_delta_v": [], "acc_chi2": [], "acc_min_cost": [], "n": n}
        for s in snrs:
            rs = [r for r in recs if r["snr"] == s]
            dv = np.array([r.get("delta_v", np.nan) for r in rs], float)
            d["mean_delta_v"].append(float(np.nanmean(np.where(np.isinf(dv), np.nan, dv))))
            d["median_delta_v"].append(float(np.nanmedian(dv)))
            d["acc_chi2"].append(float(np.mean([r["condition_hat"] == truth for r in rs])))
            mc = ["nlos" if (r.get("v_olos") is None or r["v_nlos"] <= r["v_olos"]) else "olos" for r in rs]
            d["acc_min_cost"].append(float(np.mean([m == truth for m in mc])))
        out[data_name] = d
        print(f"[unknown/{data_name}] {d}")
    out["paper_table_I"] = {"snr": [-20, -10, 0, 10], "delta_v": [5.5, 5.2, 5.0, 5.3]}
    save("unknown", out)
    return out


def _beams_sample(args):
    cfg_path, seed, G, n0, profile, known_k = args
    ov = {"beam.num_beams": int(G), "noise.mode": "n0", "noise.n0": float(n0)}
    if known_k:
        ov["estimator.num_paths"] = 1
    cfg = load_config(cfg_path, estimator=profile, **ov)
    s = MmWavePositioningSystem(cfg).simulate(seed=int(seed))
    b = compute_bounds(s)
    out = ClassicalEstimator("los")(s)
    return {"G": G, "pos_err": float(M.position_error(out.position, s.gt.position)), "peb": b.peb, "snr": s.meta.snr_db}


def exp_beams(args):
    """Constant total transmit power: unit-power beams with N0(G) = N0_ref * G / G_ref."""
    cfg_path = str(EXP / "los_area.yaml")
    G_ref, Gs = 32, [4, 8, 12, 16, 20, 24, 28, 32]
    cfg = load_config(cfg_path)
    ref = [MmWavePositioningSystem(cfg).simulate(seed=i, snr_db=args.beams_ref_snr).meta.n0 for i in range(50)]
    n0_ref = float(np.mean(ref))
    items = [(cfg_path, 3_000_000 + i, G, n0_ref * G / G_ref, args.estimator, args.known_k) for G in Gs
             for i in range(args.mc)]
    recs = pmap(_beams_sample, items, args.workers)
    out = {"G": Gs, "ref_snr_db_at_G32": args.beams_ref_snr, "n0_ref": n0_ref}
    for key in ("pos_err", "peb"):
        out[f"{key}_p90"] = [float(np.percentile([r[key] for r in recs if r["G"] == G], 90)) for G in Gs]
    out["mean_snr_db"] = [float(np.mean([r["snr"] for r in recs if r["G"] == G])) for G in Gs]
    print("[beams]", json.dumps(out))
    plots.figure_vs_beams(Gs, out["pos_err_p90"], out["peb_p90"], OUT / "beams.png")
    save("beams", out)
    return out


def exp_cdf(args):
    base = {"beam.num_beams": 20, "scenario.ue_position": {"mode": "uniform_box", "low": [2.0, 0.0], "high": [4.0, 0.3]}}
    cases = {
        "LOS": ("los.yaml", "los", {}, 1),
        "NLOS with 1 scatterer": ("nlos.yaml", "nlos", {"scenario.scatterers": {"mode": "fixed", "positions": [[1.5, 0.4]]}}, 2),
        "NLOS with 2 scatterers": ("nlos.yaml", "nlos", {"scenario.scatterers": {"mode": "fixed", "positions": [[1.5, 0.4], [1.5, 0.6]]}}, 3),
    }
    out = {}
    for label, (f, cond, ov, K) in cases.items():
        ov = _known_k(args, ov, K)
        items = [(str(EXP / f), 4_000_000 + i, -5.0) for i in range(args.mc)]
        recs = pmap(partial(run_sample, overrides={**base, **ov}, condition=cond, profile=args.estimator), items,
                    args.workers)
        out[label] = {"errors": [r["pos_err"] for r in recs], "peb": [r["peb"] for r in recs],
                      "median": float(np.median([r["pos_err"] for r in recs])),
                      "p90": float(np.percentile([r["pos_err"] for r in recs], 90))}
        print(f"[cdf] {label}: median {out[label]['median']:.4f} m, p90 {out[label]['p90']:.4f} m")
    plots.figure_cdf({k: np.array(v["errors"]) for k, v in out.items()}, "localization error [m]", OUT / "cdf.png",
                     "SNR = -5 dB, G = 20")
    save("cdf", out)
    return out


def exp_g0(args):
    cfg_path = str(EXP / "los.yaml")
    items = [(cfg_path, 5_000_000 + i, s) for s in SNRS for i in range(args.mc)]
    ov = _known_k(args, {}, 1)
    a = aggregate(pmap(partial(run_sample, overrides=ov, condition="los", profile=args.estimator), items, args.workers),
                  SNRS, 1)
    b = aggregate(pmap(partial(run_sample, overrides=ov, condition="los", oracle=True, profile=args.estimator), items,
                       args.workers), SNRS, 1)
    out = {"snr": SNRS, "dcs_somp_init": {k: a[k] for k in ("rmse_pos", "rmse_ori", "peb", "reb", "delta_pos_db")},
           "oracle_init": {k: b[k] for k in ("rmse_pos", "rmse_ori", "peb", "reb", "delta_pos_db")}}
    out["gap_db"] = [M.delta_db(x, y) for x, y in zip(a["rmse_pos"], b["rmse_pos"])]
    print("[g0] RMSE_p gap DCS-SOMP init vs oracle init [dB]:", np.round(out["gap_db"], 2))
    plots.figure_rmse_panels(SNRS, [dict(rmse=a["rmse_pos"], bound=b["rmse_pos"], label="DCS-SOMP init + SAGE",
                                         bound_label="oracle init + SAGE", ylabel="RMSE(p) [m]")],
                             "G0: initialisation gap (LOS)", OUT / "g0.png")
    save("g0", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+",
                    choices=["los", "nlos", "nlos_paper", "olos", "unknown", "beams", "cdf", "g0", "all"])
    ap.add_argument("--mc", type=int, default=200, help="Monte Carlo runs per point (paper: 1000)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--olos-top-k", type=int, default=None, help="LMA only from the k best alpha trials")
    ap.add_argument("--beams-ref-snr", type=float, default=-20.0, help="SNR at G=32 for the beam sweep")
    ap.add_argument("--estimator", default="strong_classical", help="profile in configs/estimators/")
    ap.add_argument("--unknown-mc", type=int, default=100, help="realisations per SNR for Table I (paper: 100)")
    ap.add_argument("--known-k", action="store_true", help="diagnostic: DCS-SOMP uses the true number of paths")
    args = ap.parse_args()
    global OUT
    OUT = OUT_ROOT / (args.estimator + ("_knownK" if args.known_k else ""))
    todo = ["los", "nlos", "nlos_paper", "olos", "unknown", "beams", "cdf", "g0"] if "all" in args.experiments \
        else args.experiments
    t = time.time()
    for e in todo:
        t0 = time.time()
        {"los": lambda: exp_fixed("los", "los.yaml", args, 1),
         "nlos": lambda: exp_fixed("nlos", "nlos.yaml", args, 2),
         "nlos_paper": lambda: [exp_fixed(f"nlos_los_selection_{r}_prune_{p}", "nlos.yaml", args, 2,
                                          overrides={"estimator.los_selection": r, "estimator.nlos_prune_level": p},
                                          history=())
                                for r, p in (("min_delay", None), ("min_cost", None), ("strongest", None))],
         "olos": lambda: exp_olos(args), "unknown": lambda: exp_unknown(args), "beams": lambda: exp_beams(args),
         "cdf": lambda: exp_cdf(args), "g0": lambda: exp_g0(args)}[e]()
        print(f"[{e}] done in {time.time() - t0:.0f} s")
    print(f"total {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()

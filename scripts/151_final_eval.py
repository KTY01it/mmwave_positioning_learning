"""B2f3R finalization: end-to-end evaluation of named arms (configs/experiments/b2f3r_final/arms.yaml) on identical observations
(copy of scripts/108_v3_eval.py with the v3.1 seed namespace, arms and diagnostics).

    python scripts/151_final_eval.py run    --tag T --split S --regions iid,lat --G 8,16 --n 900 --arms frozen,v2_n3 [--snr hsnr] [--cond olos]
    python scripts/151_final_eval.py timing --tag T --split S --regions iid --G 6,8,16,32 --arms ...      # clean pass, 8 processes
    python scripts/151_final_eval.py noise_only --tag T --G 6,8,16,32 --arms ...                         # pure-noise observations

Records: outputs/b2f3r_final/<tag>/<split>_<region>[_hsnr][_<cond>].jsonl, with per-path diagnostics (ground truth used for
scoring only).
"""
import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
import yaml

from _bootstrap import ROOT

BASE = ROOT / "outputs" / "b2f3r_final"
ARMS = yaml.safe_load((ROOT / "configs" / "experiments" / "b2f3r_final" / "arms.yaml").read_text())
HSNR = ("choice", [15.0, 20.0, 25.0])
AI_MIN_G = 16
_C = {}


def _est(name):
    if name not in _C:
        import torch
        torch.set_num_threads(1)
        spec = ARMS[name]
        if spec["kind"] == "fin":
            from positioning.b2f3r_final.candidate import B2f3RFinal
            _C[name] = B2f3RFinal(**dict(spec.get("kwargs", {})))
        elif spec["kind"] == "v31":
            from positioning.b2f3r_v31.candidate import B2f3Rv31
            kw = dict(spec.get("kwargs", {}))
            _C[name] = B2f3Rv31(**kw)
        elif spec["kind"] == "v3":
            from positioning.b2f3r_v3.candidate import B2f3Rv3
            kw = dict(spec.get("kwargs", {}))
            if "hyps" in kw:
                kw["hyps"] = tuple(kw["hyps"])
            _C[name] = B2f3Rv3(**kw)
        elif spec["kind"] == "b":
            from positioning.b2f3r_v2.candidate import B2f3Rv2
            _C[name] = B2f3Rv2(**spec.get("kwargs", {}))
        elif spec["kind"] == "v1":
            from positioning.b2f3r.estimator import B2f3R
            _C[name] = B2f3R(**spec["kwargs"])
        elif spec["kind"] == "kn":
            from positioning.models.v3_candidate import V3Candidate
            _C[name] = V3Candidate(it_mid=3)
        elif spec["kind"] == "a":
            from positioning.models.direct_cnn import DirectCNNEstimator
            from positioning.models.lamp import LAMPEstimator
            from positioning.models.tinynn import TinyNNEstimator
            from positioning.models.wu3dcnn import WuEstimator
            M = ROOT / "outputs" / "robustness_completion" / "r5"
            _C[name] = {"direct_cnn": lambda: DirectCNNEstimator(str(M / "direct_cnn" / "model.pt")),
                        "lamp": lambda: LAMPEstimator(str(M / "lamp" / "model.pt"), condition="unknown"),
                        "wu_3dcnn": lambda: WuEstimator(str(M / "wu3dcnn" / "model.pt")),
                        "tiny_nn": lambda: TinyNNEstimator(str(M / "tinynn" / "model.pt"))}[name]()
    return _C[name]


def run_arm(name, st, s, diag=True):
    from positioning.b2f3r_v2 import diagnostics as DG
    from positioning.robust.estimators import classical
    from positioning.robust.evalkit import score
    spec = ARMS[name]
    t0 = time.perf_counter()
    if spec["kind"] == "c":
        o, _ = classical(st, **spec.get("kwargs", {}))
    elif spec["kind"] == "a":
        o = _est(name)(st if name == "tiny_nn" else s)
    elif spec["kind"] == "kn":
        o = _est(name).estimate(st.y, dataclasses.replace(st.context, n0=st.context.n0 * spec["scale"]))
    else:
        o = _est(name).estimate(st.y, st.context)
    r = score(o, s, time.perf_counter() - t0)
    lg = o.info.get("loop", {})
    r.update({k: lg[k] for k in ("n_sage", "sweeps", "rounds", "switched", "lobe_rerun", "support_repair", "stop_ok",
                                 "level_evals", "fwd_entered", "accepted") if k in lg})
    if "timing_v2" in o.info:
        r["tm"] = o.info["timing_v2"]
    if "v3" in o.info:
        v = o.info["v3"]
        r["v3"] = {"trig": v.get("triggered", 0), "sel": v.get("selected"), "amb": v.get("ambiguous", 0),
                   "margin": v.get("margin"), "skip": v.get("skipped"), "gen": v.get("generated"), "n_hyps": v.get("n_hyps"),
                   "n_loc": v.get("n_loc"), "gate": v.get("gate"), "t": v.get("t")}
    if "geo" in o.info:
        r["geo"] = o.info["geo"]
    r["c_hat"] = o.info.get("c_hat", o.condition)
    if diag and getattr(o, "channel", None) is not None and spec["kind"] in ("fin", "v31", "v3", "b", "v1", "c", "kn"):
        pr = DG.path_report(o.channel, s)
        r["path"] = {"snr": [round(v, 2) for v in pr["snr_db"]], "m": pr["matched"], "ok": pr.get("support_ok", 0)}
    return r


def work(args):
    from positioning.b2f3r_final import data as D
    from positioning.evaluation.benchmark import with_estimator_profile
    from positioning.robust.evalkit import bounds
    from positioning.system.simulator import subsample_beams
    split, region, i, G_list, arms, hs, strong_n, cond = args
    s32, _ = D.draw(split, i, snr=HSNR if hs else ("uniform", -10.0, 10.0), region=region, condition=cond)
    if s32 is None:
        return []
    out = []
    for G in G_list:
        s = subsample_beams(s32, G)
        st = with_estimator_profile(s, "strong_classical")
        peb, reb = bounds(s)
        r = {"split": split, "region": region, "index": i, "G": G, "snr": float(s.meta.snr_db), "condition": s.gt.condition,
             "K": int(s.gt.num_paths), "peb": peb, "reb": reb, "n0": float(s.context.n0), "sep": D.separation(s),
             "pos": [float(v) for v in s.gt.position], "alpha": float(s.gt.orientation), "arms": {}}
        for a in arms:
            if (a.startswith("strong") and i >= strong_n) or (ARMS[a]["kind"] == "a" and G < AI_MIN_G):
                continue
            r["arms"][a] = run_arm(a, st, s)
        out.append(r)
    return out


def path_of(a, region):
    return BASE / a.tag / f"{a.split}_{region}{'_hsnr' if a.snr == 'hsnr' else ''}{'_' + a.cond if a.cond else ''}.jsonl"


def run(a, timing=False):
    (BASE / a.tag).mkdir(parents=True, exist_ok=True)
    for region in a.regions:
        p = path_of(a, region)
        if timing:
            p = p.with_name(p.stem + "_timing.jsonl")
        assert not p.exists(), f"{p} exists"
        split = f"{a.split}_{region}{'_hsnr' if a.snr == 'hsnr' else ''}{'_' + a.cond if a.cond else ''}"
        items = [(split, region, i, a.G, a.arms, a.snr == "hsnr", a.strong_n, a.cond) for i in range(a.timing_n if timing else a.n)]
        t0 = time.time()
        with ProcessPoolExecutor(8 if timing else a.workers) as ex, open(p, "w") as fh:
            for recs in ex.map(work, items, chunksize=1 if timing else 2):
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        print(p.name, "done", f"{time.time() - t0:.0f}s", flush=True)


def noise_only_work(args):
    """Pure-noise observation (y = noise with the sample's N0): calibration of the stop under H0."""
    from positioning.b2f3r_final import data as D
    from positioning.evaluation.benchmark import with_estimator_profile
    from positioning.system.simulator import subsample_beams
    i, G_list, arms, split = args
    s32, sd = D.draw(split, i)
    rng = np.random.default_rng([sd["noise"] % (2 ** 32), 7])
    out = []
    for G in G_list:
        st = with_estimator_profile(subsample_beams(s32, G), "strong_classical")
        w = (rng.standard_normal(st.y.shape) + 1j * rng.standard_normal(st.y.shape)) * np.sqrt(0.5 * st.context.n0)
        z = dataclasses.replace(st, y=w)
        r = {"index": i, "G": G, "arms": {}}
        for a in arms:
            t0 = time.perf_counter()
            o = _est(a).estimate(z.y, z.context)
            lg = o.info.get("loop", {})
            r["arms"][a] = {"stop_rejects_at_start": int(lg.get("fwd_entered", 0) > 0) if "fwd_entered" in lg else None,
                            "accepted": int(lg.get("accepted", 0)), "K": int(o.channel.num_paths) if o.channel is not None else None,
                            "n0_ratio": float(o.info.get("n0_used", np.nan) / st.context.n0), "t": time.perf_counter() - t0,
                            "switched": int(lg.get("switched", 0)), "fwd": int(lg.get("fwd_entered", 0))}
        out.append(r)
    return out


def noise_only(a):
    (BASE / a.tag).mkdir(parents=True, exist_ok=True)
    p = BASE / a.tag / "noise_only.jsonl"
    assert not p.exists()
    with ProcessPoolExecutor(a.workers) as ex, open(p, "w") as fh:
        for recs in ex.map(noise_only_work, [(i, a.G, a.arms, f"{a.split}_noise_only") for i in range(a.n)], chunksize=4):
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    print("noise_only done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "timing", "noise_only"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="fint")
    ap.add_argument("--regions", type=lambda s: s.split(","), default=["iid"])
    ap.add_argument("--G", type=lambda s: tuple(int(v) for v in s.split(",")), default=(16, 32))
    ap.add_argument("--arms", type=lambda s: s.split(","), required=True)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--snr", choices=["main", "hsnr"], default="main")
    ap.add_argument("--cond", default=None)
    ap.add_argument("--strong-n", dest="strong_n", type=int, default=10 ** 9)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--timing-n", dest="timing_n", type=int, default=200)
    a = ap.parse_args()
    {"run": run, "timing": lambda a: run(a, timing=True), "noise_only": noise_only}[a.mode](a)

"""P8: propagation-condition routing for the gated proposed method (protocol docs/p8_protocol.md, locked).

R1 (classical): Sec. IV-D test of Shahmansoori et al. [R1] in its chi-square form (F19, strong v2.2) applied to the
    fast-path channel estimate: NLOS (LOS when K = 1) unless the unpruned NLOS EXIP cost is rejected, else OLOS.
R2 (learned): CondNet, P(c | Y) from the angle-delay image (the ADCPM-like input of the path-init network), in the
    spirit of learned NLOS identification from ADCPM [R20], [R5]; the architecture is the project's DESIGN.

RoutedGatedEstimator: Y -> shared fast stages -> c_hat -> branch (LOS / NLOS / OLOS) -> gate v1.1 (frozen thresholds)
    -> fallback strong classical v2.2 in unknown-condition mode (never uses c_hat).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import chi2
from torch import nn

from ..localization.exip_lma import exip_lma, pack, weight_matrix
from ..localization.nlos import nlos_localize
from ..localization.pipeline import ClassicalEstimator, localize
from ..signal_processing.model import ChannelEstimate, ObservationModel
from ..signal_processing.sage import sage
from .base import Estimator, PositionEstimate
from .gated import _strong_context, load_locked_gate, thresholds_for
from .gated_v11 import valid_fast_output
from .geom_init import GeomInitializer
from .path_init import PathInitEstimator, decode, ls_gains, network_input, validate_paths

CONDITIONS = ("los", "nlos", "olos")


# ---------------------------------------------------------------------------------------------------------- network
class CondNet(nn.Module):
    """Small CNN classifier on the [21, 65, 65] path-init input -> 3 logits (LOS, NLOS, OLOS)."""

    def __init__(self, in_ch: int = 21, width: int = 16):
        super().__init__()
        w = width

        def block(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True))
        self.features = nn.Sequential(block(in_ch, w, 1), block(w, 2 * w, 2), block(2 * w, 4 * w, 2),
                                      block(4 * w, 4 * w, 2), block(4 * w, 4 * w, 2))   # 65 -> 33 -> 17 -> 9 -> 5
        self.head = nn.Sequential(nn.Linear(8 * w, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.head(f)


class LearnedRouter:
    def __init__(self, checkpoint: str):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.net = CondNet(**ck["model_kwargs"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.net(torch.from_numpy(x)[None]), dim=1)[0].numpy()

    def route(self, x: np.ndarray) -> tuple[str, np.ndarray]:
        p = self.probabilities(x)
        return CONDITIONS[int(np.argmax(p))], p


# -------------------------------------------------------------------------------------------- classical test (R1)
def classical_route(est: ChannelEstimate, context) -> tuple[str, dict]:
    """Sec. IV-D chi-square test on the given channel estimate (same weighting / LOS selection / level as strong v2.2)."""
    e = context.config.estimator
    K = est.num_paths
    if K <= 1:
        return "los", {"p_nlos": 1.0, "v_nlos": 0.0}
    nl = nlos_localize(est, context, e.exip_weighting, e.los_selection, None)   # unpruned, as in unknown.py
    p = float(chi2.sf(nl.cost, K - 1))
    return ("nlos" if p >= e.unknown_test_level else "olos"), {"p_nlos": p, "v_nlos": float(nl.cost)}


# ------------------------------------------------------------------------------------------- shared fast stages
def fast_channel_estimate(pe: PathInitEstimator, x: np.ndarray, y, context):
    """PathInitEstimator.channel_estimate with the network input x given (computed once, shared with the router)."""
    cfg = context.config
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pe.net(torch.from_numpy(x)[None])[0].numpy()
    t_net = time.perf_counter()
    dec = decode(out, cfg, pe.threshold, cfg.estimator.max_paths)
    Ts = cfg.physics.sampling_period
    delay = np.array([d[3] * Ts for d in dec])
    aod = np.array([d[1] for d in dec])
    aoa = np.array([d[2] for d in dec])
    gain = ls_gains(ObservationModel(context), y, delay, aod, aoa)
    init = ChannelEstimate(delay, aod, aoa, gain, info={"probabilities": [d[0] for d in dec]})
    t1 = time.perf_counter()
    b = cfg.estimator.sage_max_iter if pe.sage_iterations is None else pe.sage_iterations
    est = sage(y, context, init, max_iter=b, tol=cfg.estimator.sage_tol).estimate if b > 0 else init
    t2 = time.perf_counter()
    if pe.validate and est.num_paths > 1:
        est = validate_paths(y, context, est, cfg.estimator.min_paths)
    t3 = time.perf_counter()
    return est, {"path_net": t_net - t0, "decode_ls": t1 - t_net, "sage": t2 - t1, "validation": t3 - t2}


def branch_localize(est: ChannelEstimate, y, context, cond: str, geom: GeomInitializer) -> PositionEstimate:
    """The localisation + gate-score part of ProposedEstimator.estimate for an explicit condition."""
    cfg = context.config
    if cond == "olos" and est.num_paths >= 2:
        e = est.sorted_by_delay()
        c = cfg.physics.speed_of_light
        p0, a0, s0 = geom.predict(e, c)
        r = exip_lma(e, weight_matrix(e, context, cfg.estimator.exip_weighting), pack(p0, a0, s0, e.gain_tilde, False),
                     False, context.bs_position, c)
        out = PositionEstimate(r.position, r.orientation, "olos", r.scatterers, e, r.cost,
                               info={"lma_nfev": r.nfev, "lma_success": r.success})
    else:
        out = localize(est, context, cond)
    ch = out.channel if out.channel is not None else est
    M = y.size
    res = float(np.sum(np.abs(y - ObservationModel(context).mean(ch)) ** 2) / context.n0)
    out.info["gate"] = {"r_norm": res / M, "z": (res - M) / np.sqrt(M),
                        "v": float(out.cost) if np.isfinite(out.cost) else 0.0}
    return out


def gate_accept(out: PositionEstimate, G: int, variant: str, table: dict) -> bool:
    """Gate v1.1: validity check, then the frozen 6C thresholds."""
    if not valid_fast_output(out.position):
        return False
    th = thresholds_for(G, table)
    z_ok, v_ok = out.info["gate"]["z"] <= th["z"], out.info["gate"]["v"] <= th["v"]
    return {"z": z_ok, "v": v_ok, "zv": z_ok and v_ok}[variant]


# --------------------------------------------------------------------------------------------- routed estimator
class RoutedGatedEstimator(Estimator):
    requires_oracle = False

    def __init__(self, path_ckpt: str, geom_ckpt: str, router: str, router_ckpt: str | None = None):
        if router not in ("learned", "classical", "oracle"):
            raise ValueError(router)
        self.paths = PathInitEstimator(path_ckpt, sage_iterations=1)
        self.geom = GeomInitializer(geom_ckpt)
        self.router = router
        self.learned = LearnedRouter(router_ckpt) if router == "learned" else None
        self.variant, self.table = load_locked_gate()
        self.requires_oracle = router == "oracle"
        self.name = f"routed_gated_{router}"

    def estimate(self, y, context, gt=None) -> PositionEstimate:
        t0 = time.perf_counter()
        x = network_input(y, context)
        est, stages = fast_channel_estimate(self.paths, x, y, context)
        tr = time.perf_counter()
        if self.router == "learned":
            cond, probs = self.learned.route(x)
            rinfo = {"probabilities": probs.tolist()}
        elif self.router == "classical":
            cond, rinfo = classical_route(est, context)
        else:
            cond, rinfo = context.config.scenario.condition, {}
        t_router = time.perf_counter() - tr
        out = branch_localize(est, y, context, cond, self.geom)
        t_fast = time.perf_counter() - t0
        info = {"c_hat": cond, "router": rinfo, "t_router": t_router, "t_fast": t_fast, "stages": stages}
        if gate_accept(out, y.shape[1], self.variant, self.table):
            out.info["routed"] = {**info, "accepted": True}
            return out
        fb_cond = context.config.scenario.condition if self.router == "oracle" else "unknown"
        t1 = time.perf_counter()
        fb = ClassicalEstimator(fb_cond).estimate(y, _strong_context(context))
        fb.info["routed"] = {**info, "accepted": False, "t_fallback": time.perf_counter() - t1}
        return fb

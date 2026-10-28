"""All geometric relations of the problem (Shahmansoori et al., Eqs. (17)-(22), (26), (61)-(65)).

This is the ONLY module that contains geometry formulas. Channel, estimators and bounds import
from here.

Conventions (docs/P0_specification.md, F1/F2/F16):
* angles are measured counter-clockwise from the +x axis, wrapped to (-pi, pi];
* AoD  theta_Tx,k = direction BS -> first interaction point (MS for LOS, s_k for NLOS);
* AoA  theta_Rx,k = (direction MS -> last interaction point) - alpha, i.e. in the MS array frame;
* LOS: theta_Rx,0 = theta_Tx,0 + pi - alpha (Eq. 22 written with atan2 instead of arccos).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def wrap_angle(x):
    """Wrap angle(s) to (-pi, pi]."""
    y = np.pi - np.mod(np.pi - np.asarray(x, dtype=float), 2 * np.pi)
    return y if y.ndim else float(y)


def unit(phi):
    """Unit vector(s) [cos phi, sin phi] with shape phi.shape + (2,)."""
    phi = np.asarray(phi, dtype=float)
    return np.stack([np.cos(phi), np.sin(phi)], axis=-1)


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(2)


# --------------------------------------------------------------------------------------
# forward geometry
# --------------------------------------------------------------------------------------
def compute_los_geometry(bs_pos, ue_pos, alpha, c) -> dict:
    """Eqs. (17), (19), (22)."""
    q, p = _vec(bs_pos), _vec(ue_pos)
    d = float(np.linalg.norm(p - q))
    aod = float(np.arctan2(p[1] - q[1], p[0] - q[0]))
    return {
        "delay": d / c,
        "aod": float(wrap_angle(aod)),
        "aoa": float(wrap_angle(aod + np.pi - alpha)),
        "distance": d,
    }


def compute_nlos_geometry(bs_pos, ue_pos, scatterer_pos, alpha, c) -> dict:
    """Eqs. (18), (20), (21) for a single-bounce path via scatterer s."""
    q, p, s = _vec(bs_pos), _vec(ue_pos), _vec(scatterer_pos)
    d1 = float(np.linalg.norm(s - q))
    d2 = float(np.linalg.norm(p - s))
    aod = float(np.arctan2(s[1] - q[1], s[0] - q[0]))
    aoa = float(np.arctan2(s[1] - p[1], s[0] - p[0]) - alpha)
    return {
        "delay": (d1 + d2) / c,
        "aod": float(wrap_angle(aod)),
        "aoa": float(wrap_angle(aoa)),
        "distance": d1 + d2,
        "d1": d1,
        "d2": d2,
    }


@dataclass
class PathGeometry:
    """Geometry of all K paths. Index 0 is the LOS path when ``has_los``."""

    delay: np.ndarray
    aod: np.ndarray
    aoa: np.ndarray
    distance: np.ndarray
    d2: np.ndarray  # MS-side segment length (= distance for LOS)
    is_los: np.ndarray  # bool per path
    scatterers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def num_paths(self) -> int:
        return int(self.delay.size)

    @property
    def has_los(self) -> bool:
        return bool(self.is_los.any())


def compute_path_geometry(bs_pos, ue_pos, alpha, scatterers, include_los: bool, c) -> PathGeometry:
    """Delays/angles of the LOS path (optional) followed by one path per scatterer."""
    scatterers = np.asarray(scatterers, dtype=float).reshape(-1, 2)
    rows = []
    if include_los:
        g = compute_los_geometry(bs_pos, ue_pos, alpha, c)
        rows.append((g["delay"], g["aod"], g["aoa"], g["distance"], g["distance"], True))
    for s in scatterers:
        g = compute_nlos_geometry(bs_pos, ue_pos, s, alpha, c)
        rows.append((g["delay"], g["aod"], g["aoa"], g["distance"], g["d2"], False))
    if not rows:
        raise ValueError("No paths: include_los=False and no scatterers")
    cols = list(zip(*rows))
    return PathGeometry(
        delay=np.array(cols[0]),
        aod=np.array(cols[1]),
        aoa=np.array(cols[2]),
        distance=np.array(cols[3]),
        d2=np.array(cols[4]),
        is_los=np.array(cols[5], dtype=bool),
        scatterers=scatterers.copy(),
    )


# --------------------------------------------------------------------------------------
# literal paper formulas (arccos form), used only by Level-1 tests to document F1
# --------------------------------------------------------------------------------------
def paper_geometry_arccos(bs_pos, ue_pos, alpha, scatterer_pos, c) -> dict:
    """Eqs. (17)-(22) exactly as printed (valid only for p_y >= q_y and s_y >= p_y)."""
    q, p, s = _vec(bs_pos), _vec(ue_pos), _vec(scatterer_pos)
    return {
        "tau0": np.linalg.norm(p - q) / c,
        "tauk": np.linalg.norm(q - s) / c + np.linalg.norm(p - s) / c,
        "aod0": np.arccos((p[0] - q[0]) / np.linalg.norm(p - q)),
        "aodk": np.arccos((s[0] - q[0]) / np.linalg.norm(s - q)),
        "aoak": np.pi - np.arccos((p[0] - s[0]) / np.linalg.norm(p - s)) - alpha,
        "aoa0": np.pi + np.arccos((p[0] - q[0]) / np.linalg.norm(p - q)) - alpha,
    }


# --------------------------------------------------------------------------------------
# Jacobian of channel parameters w.r.t. position-domain parameters (Eqs. 16, 23-26)
# --------------------------------------------------------------------------------------
# Channel parameter ordering per path (Eq. 8): [tau, theta_Tx, theta_Rx, Re h~, Im h~]
CHANNEL_PARAMS_PER_PATH = 5


def geometry_parameter_layout(num_paths: int, has_los: bool) -> list[str]:
    """Names of the position-domain parameters eta_bar (Sec. III-B).

    has_los:  [px, py, alpha, h0R, h0I, s1x, s1y, h1R, h1I, ...]          (4K+5 for K scatterers)
    OLOS:     [px, py, alpha, s1x, s1y, h1R, h1I, ...]                     (eta_bar_olos)
    """
    names = ["px", "py", "alpha"]
    for k in range(num_paths):
        if has_los and k == 0:
            names += ["h0R", "h0I"]
        else:
            names += [f"s{k}x", f"s{k}y", f"h{k}R", f"h{k}I"]
    return names


def transformation_matrix(bs_pos, ue_pos, alpha, scatterers, has_los: bool, c) -> np.ndarray:
    """T = d eta^T / d eta_bar  (shape [len(eta_bar), 5K]), Eq. (16).

    Derivatives are derived from the atan2 forms (F2) and verified against finite differences
    in tests/test_geometry.py.
    """
    q, p = _vec(bs_pos), _vec(ue_pos)
    scatterers = np.asarray(scatterers, dtype=float).reshape(-1, 2)
    K = int(has_los) + scatterers.shape[0]
    layout = geometry_parameter_layout(K, has_los)
    idx = {n: i for i, n in enumerate(layout)}
    T = np.zeros((len(layout), CHANNEL_PARAMS_PER_PATH * K))

    def col(k, j):
        return CHANNEL_PARAMS_PER_PATH * k + j

    k = 0
    if has_los:
        d0 = np.linalg.norm(p - q)
        th = np.arctan2(p[1] - q[1], p[0] - q[0])
        dtau_dp = unit(th) / c
        dth_dp = np.array([-np.sin(th), np.cos(th)]) / d0
        T[idx["px"]:idx["py"] + 1, col(0, 0)] = dtau_dp
        T[idx["px"]:idx["py"] + 1, col(0, 1)] = dth_dp
        T[idx["px"]:idx["py"] + 1, col(0, 2)] = dth_dp  # theta_Rx,0 = theta_Tx,0 + pi - alpha
        T[idx["alpha"], col(0, 2)] = -1.0
        T[idx["h0R"], col(0, 3)] = 1.0
        T[idx["h0I"], col(0, 4)] = 1.0
        k = 1
    for j, s in enumerate(scatterers):
        kk = k + j
        sx, sy, hR, hI = (idx[f"s{kk}x"], idx[f"s{kk}y"], idx[f"h{kk}R"], idx[f"h{kk}I"])
        d1 = np.linalg.norm(s - q)
        d2 = np.linalg.norm(s - p)
        th_t = np.arctan2(s[1] - q[1], s[0] - q[0])
        phi = np.arctan2(s[1] - p[1], s[0] - p[0])  # = theta_Rx,k + alpha
        u_t, u_r = unit(th_t), unit(phi)
        # tau_k = (|s-q| + |s-p|)/c
        T[idx["px"]:idx["py"] + 1, col(kk, 0)] = -u_r / c
        T[sx:sy + 1, col(kk, 0)] = (u_t + u_r) / c
        # theta_Tx,k depends on s only
        T[sx:sy + 1, col(kk, 1)] = np.array([-np.sin(th_t), np.cos(th_t)]) / d1
        # theta_Rx,k = atan2(s-p) - alpha
        g = np.array([-np.sin(phi), np.cos(phi)]) / d2
        T[sx:sy + 1, col(kk, 2)] = g
        T[idx["px"]:idx["py"] + 1, col(kk, 2)] = -g
        T[idx["alpha"], col(kk, 2)] = -1.0
        T[hR, col(kk, 3)] = 1.0
        T[hI, col(kk, 4)] = 1.0
    return T


# --------------------------------------------------------------------------------------
# inverse geometry (estimators): Eqs. (61)-(62), NLOS init, OLOS linear system (65)
# --------------------------------------------------------------------------------------
def los_position_from_parameters(delay, aod, aoa, bs_pos, c):
    """Eqs. (61)-(62): p = q + c tau [cos aod, sin aod], alpha = pi + aod - aoa."""
    p = _vec(bs_pos) + c * float(delay) * unit(float(aod))
    alpha = wrap_angle(np.pi + aod - aoa)
    return p, float(alpha)


def intersect_lines(p1, phi1, p2, phi2):
    """Intersection of the lines p1 + t1 u(phi1) and p2 + t2 u(phi2). Returns (point, t1, t2)."""
    A = np.column_stack([unit(phi1), -unit(phi2)])
    t = np.linalg.lstsq(A, _vec(p2) - _vec(p1), rcond=None)[0]
    return _vec(p1) + t[0] * unit(phi1), float(t[0]), float(t[1])


def scatterer_from_lines(bs_pos, ue_pos, alpha, aod, aoa):
    """NLOS init (Sec. IV-D): intersect BS ray (angle aod) with MS ray (angle aoa + alpha)."""
    s, _, _ = intersect_lines(bs_pos, aod, ue_pos, aoa + alpha)
    return s


def olos_linear_solve(bs_pos, alpha, delays, aods, aoas, c):
    """Eq. (65) for two NLOS paths (F16 sign convention).

    p = q + d_{k,1} u(theta_Tx,k) - (c tau_k - d_{k,1}) u(theta_Rx,k + alpha),  k in {k1, k2}
    Unknowns [px, py, d_{1,1}, d_{2,1}]. Returns (p, d1 array, scatterers [2,2]).
    """
    q = _vec(bs_pos)
    A = np.zeros((4, 4))
    b = np.zeros(4)
    for j in range(2):
        ut = unit(aods[j])
        ur = unit(aoas[j] + alpha)
        rows = slice(2 * j, 2 * j + 2)
        A[rows, 0:2] = np.eye(2)
        A[rows, 2 + j] = -(ut + ur)
        b[rows] = q - c * delays[j] * ur
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    p = sol[:2]
    d1 = sol[2:]
    s = np.stack([q + d1[j] * unit(aods[j]) for j in range(2)])
    return p, d1, s

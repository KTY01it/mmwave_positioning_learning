"""Typed configuration for the physical system and the estimators.

A YAML config may contain ``extends: <relative path>``; the parent is loaded first and the
child is deep-merged on top. All quantities are SI (m, s, Hz, rad) after loading.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicsConfig:
    speed_of_light: float = 299_792_000.0  # paper: c = 0.299792 m/ns
    fc: float = 60e9
    bandwidth: float = 100e6
    num_subcarriers: int = 20
    # lambda_n = c / (n/(N Ts) + fc) (Eq. 5). False -> lambda_n = lambda_c (narrowband, GitHub repo)
    frequency_dependent_array: bool = True

    @property
    def sampling_period(self) -> float:
        return 1.0 / self.bandwidth

    @property
    def wavelength_c(self) -> float:
        return self.speed_of_light / self.fc


@dataclass(frozen=True)
class ArrayConfig:
    num_tx: int = 65
    num_rx: int = 65
    spacing_wavelengths: float = 0.5  # d = 0.5 lambda_c
    index_origin: str = "centered"  # "centered" (paper Eq. 5) | "zero" (getResponse.m)


@dataclass(frozen=True)
class BeamConfig:
    num_beams: int = 32  # G
    num_streams: int = 1  # Mt
    analog_frequency_flat: bool = True  # paper: F_RF does not depend on n
    normalize: bool = True  # ||F^{(g)}[n]||_F = 1
    symbols: str = "random_phase"  # "random_phase" (paper) | "ones" (GitHub: symbols folded into F)


@dataclass(frozen=True)
class ChannelConfig:
    gain_model: str = "pathloss"  # "pathloss" (Eqs. 66-67) | "constant" (GitHub h=10)
    constant_gain: float = 10.0  # value of h_k / sqrt(rho_k) in "constant" mode
    gain_phase: str = "random"  # "random" | "zero"
    reflection_loss_db_mean: float = -10.0  # sigma_0^2
    reflection_loss_db_std: float = 4.0
    reflection_loss_random: bool = True
    atmospheric_attenuation_db_per_km: float = 16.0
    poisson_density: float = 1.0 / 7.0  # gamma_r


@dataclass(frozen=True)
class SamplerConfig:
    """Fixed value or random draw. mode in {fixed, uniform_box, uniform, none}."""

    mode: str = "fixed"
    value: Any = None
    low: Any = None
    high: Any = None
    num: int = 0
    num_max: int | None = None  # scatterers: number uniform in {num, ..., num_max}
    positions: Any = None
    # scatterers: "back" keeps only scatterers behind the MS array, (s - p) . u(alpha) < 0, so that every AoA lies in
    # the half-plane the estimators assume (F27). None = no restriction (original draw order).
    half_plane: str | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    dimension: int = 2
    bs_position: tuple = (0.0, 0.0)
    condition: str = "los"  # "los" | "nlos" | "olos"
    ue_position: SamplerConfig = field(default_factory=lambda: SamplerConfig(mode="fixed", value=(4.0, 0.0)))
    orientation: SamplerConfig = field(default_factory=lambda: SamplerConfig(mode="fixed", value=0.1))
    scatterers: SamplerConfig = field(default_factory=lambda: SamplerConfig(mode="none"))


@dataclass(frozen=True)
class NoiseConfig:
    mode: str = "target_snr"  # "target_snr" (Eq. 68, realised signal energy) | "n0"
    snr_db: float = 0.0
    n0: float | None = None


@dataclass(frozen=True)
class EstimatorConfig:
    # beamspace dictionary
    dictionary_oversampling: int = 1
    dictionary_grid: str = "centered"  # "centered" (paper) | "github"
    dictionary_normalize: bool = True
    aoa_branch: str = "back"  # MS array faces the BS (main.m L109)
    aod_branch: str = "front"
    # DCS-SOMP. "dcs_somp": Algorithm 1 literally (grid atoms). "dcs_somp_offgrid": each selected grid atom
    # is refined to continuous angles before the residual update, so that the threshold (41) is not
    # triggered by grid leakage (F17).
    detection: str = "dcs_somp_offgrid"
    pfa: float = 1e-3
    # Continuous-angle maximisation raises the noise-only statistic: the threshold (41) is evaluated with
    # factor * Nt*Nr effective columns (calibrated by scripts/calibrate_detection_threshold.py: ~4.2-5.2).
    offgrid_threshold_factor: float = 5.0
    num_paths: int | None = None  # fixed number of atoms (GitHub mode); None -> threshold (41)
    max_paths: int = 8
    min_paths: int = 1  # always accept this many atoms (the paper reports estimates even at -30 dB)
    # delay
    delay_method: str = "periodogram"  # "periodogram" (Eq. 54) | "phase_slope" (main.m)
    delay_oversampling: int = 16
    # SAGE
    sage_enabled: bool = True
    sage_max_iter: int = 20
    sage_tol: float = 1e-7
    # position
    exip_weighting: str = "fim"  # "fim" | "identity"
    # NLOS: which path is the LOS. "min_delay" = paper (smallest delay); "min_cost" = run EXIP with every path as
    # LOS and keep the smallest cost; "strongest" = largest |h~| (F21: two-path role-swap ambiguity)
    los_selection: str = "strongest"
    # NLOS: drop geometrically inconsistent paths while the EXIP cost fails a chi2(K-1) test at this level
    # (None = paper, keep every detected path; F24)
    nlos_prune_level: float | None = 1e-3
    olos_alpha_step: float = 0.05
    olos_alpha_max: float = 0.5
    # path pair used in the linear initialisation (65): "all" pairs (best cost kept) | "first_two" (two smallest delays)
    olos_pairs: str = "all"
    # unknown condition: "min_cost" = paper (smaller of (63)/(64)); "chi2" = accept NLOS unless its FIM-weighted
    # cost is rejected by a chi-square test (v ~ chi2(5K - #unknowns) under the true hypothesis). Needed because
    # with K = 3 paths the OLOS problem is exactly determined, so v_olos -> 0 and min_cost always picks OLOS.
    unknown_rule: str = "chi2"
    unknown_test_level: float = 0.01
    olos_lma_top_k: int | None = None
    # multi-start SAGE (P6 strong-init baseline): number of AoD lobes of the strongest path tried; 0 = off
    multistart: int = 0
    # optional geometric prior for selecting among multi-start candidates (the same prior an AI model would learn):
    # {"p_low": [x, y], "p_high": [x, y], "alpha_max": a, "margin": m}; None = pure maximum likelihood
    prior_box: Any = None  # None: LMA for every alpha trial (paper); k: only the k best initial costs


@dataclass(frozen=True)
class SystemConfig:
    name: str = "unnamed"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    array: ArrayConfig = field(default_factory=ArrayConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)

    @property
    def antenna_spacing(self) -> float:
        """Absolute element spacing d [m]."""
        return self.array.spacing_wavelengths * self.physics.wavelength_c

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **overrides) -> "SystemConfig":
        """Return a copy with dotted-key overrides, e.g. replace(**{"beam.num_beams": 8})."""
        d = self.to_dict()
        for key, val in overrides.items():
            _set_dotted(d, key, val)
        return config_from_dict(d)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
_SECTIONS = {
    "physics": PhysicsConfig,
    "array": ArrayConfig,
    "beam": BeamConfig,
    "channel": ChannelConfig,
    "noise": NoiseConfig,
    "estimator": EstimatorConfig,
}


def _set_dotted(d: dict, key: str, val) -> None:
    parts = key.replace("__", ".").split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val


def _deep_merge(base: dict, child: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _to_tuple(v):
    """Lists -> tuples (hashable, frozen); numeric strings such as '60e9' -> float.

    PyYAML follows YAML 1.1, which reads ``60e9`` (no dot / no exponent sign) as a string.
    """
    if isinstance(v, list):
        return tuple(_to_tuple(x) for x in v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _make(cls, d: dict | None):
    d = dict(d or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _to_tuple(v) for k, v in d.items()})


def config_from_dict(d: dict) -> SystemConfig:
    d = dict(d)
    d.pop("extends", None)
    kwargs: dict[str, Any] = {"name": d.pop("name", "unnamed")}
    for sec, cls in _SECTIONS.items():
        kwargs[sec] = _make(cls, d.pop(sec, None))
    sc = dict(d.pop("scenario", {}) or {})
    for key in ("ue_position", "orientation", "scatterers"):
        if key in sc and not isinstance(sc[key], SamplerConfig):
            sc[key] = _make(SamplerConfig, sc[key])
    kwargs["scenario"] = ScenarioConfig(**{k: _to_tuple(v) if not isinstance(v, SamplerConfig) else v
                                           for k, v in sc.items()})
    if d:
        raise KeyError(f"Unknown top-level config keys: {sorted(d)}")
    return SystemConfig(**kwargs)


def _load_raw(path: Path) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.get("extends")
    if parent:
        base = _load_raw((path.parent / parent).resolve())
        raw = _deep_merge(base, raw)
    raw.pop("extends", None)
    return raw


ESTIMATOR_DIR_NAME = "estimators"


def _profile_path(profile: str | Path) -> Path:
    p = Path(profile)
    return p.resolve() if p.suffix in (".yaml", ".yml") else CONFIG_DIR / ESTIMATOR_DIR_NAME / f"{profile}.yaml"


def load_config(path: str | Path, estimator: str | Path | None = None, **overrides) -> SystemConfig:
    """Load a YAML config (following ``extends``) and apply dotted-key overrides.

    Estimator settings are layered: profile (``estimator`` argument, else the config's ``estimator_profile``,
    i.e. configs/estimators/<name>.yaml) -> the config's own ``estimator:`` keys (experiment-specific, e.g. the
    OLOS alpha step) -> dotted overrides.
    """
    raw = _load_raw(Path(path).resolve())
    profile = estimator if estimator is not None else raw.get("estimator_profile")
    raw.pop("estimator_profile", None)
    if profile is not None:
        prof = yaml.safe_load(_profile_path(profile).read_text()) or {}
        raw["estimator"] = _deep_merge(prof.get("estimator", {}), raw.get("estimator", {}) or {})
        raw["name"] = f"{raw.get('name', 'unnamed')}@{prof.get('name', Path(str(profile)).stem)}"
    for key, val in overrides.items():
        _set_dotted(raw, key, val)
    return config_from_dict(raw)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

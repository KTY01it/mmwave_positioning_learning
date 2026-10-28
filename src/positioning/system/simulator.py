"""Orchestration of the physical system. No mathematics here; see the individual modules.

config -> scenario (p, alpha, s_k) -> geometry (tau, AoD, AoA) -> gains -> H[n]
       -> beams F, symbols x -> noiseless HFx -> AWGN (Eq. 68) -> y, + ground truth.

Every random component has its own seed (geometry, gain, beam, noise) so that e.g. a test split can
hold geometry and beams fixed while drawing fresh noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import beamforming, channel, geometry, noise, observation
from .arrays import subcarrier_wavelengths
from .scenario import sample_scenario

SEED_COMPONENTS = ("geometry", "gain", "beam", "noise")


@dataclass
class GroundTruth:
    position: np.ndarray
    orientation: float
    condition: str
    paths: channel.PathParameters
    scatterers: np.ndarray
    bs_position: np.ndarray

    @property
    def num_paths(self) -> int:
        return self.paths.num_paths


@dataclass
class Metadata:
    snr_db: float  # Eq. (68) for this realisation (target, or implied by N0 in mode "n0")
    n0: float
    seed: int | None
    seed_geometry: int
    seed_gain: int
    seed_beam: int
    seed_noise: int
    config_name: str
    measured_snr_db: float = float("nan")


@dataclass
class ObservationContext:
    """Everything the receiver legitimately knows besides y: system config, pilots, N0."""

    config: object
    F: np.ndarray  # [N, G, Nt, Mt]
    x: np.ndarray  # [N, G, Mt]
    n0: float
    bs_position: np.ndarray
    wavelengths: np.ndarray  # [N]

    @property
    def s(self) -> np.ndarray:
        """Effective transmitted vectors F x, [N, G, Nt]."""
        return beamforming.effective_beams(self.F, self.x)


@dataclass
class SimulationOutput:
    y: np.ndarray  # [N, G, Nr] — the only signal an estimator sees
    h: np.ndarray  # [N, Nr, Nt] — channel, for validation only
    gt: GroundTruth
    meta: Metadata
    context: ObservationContext
    noiseless: np.ndarray = field(repr=False, default=None)  # H F x, for validation only

    # aliases matching the API sketch
    @property
    def geometry(self) -> dict:
        return {"bs_position": self.gt.bs_position, "ue_position": self.gt.position,
                "orientation": self.gt.orientation, "scatterers": self.gt.scatterers,
                "condition": self.gt.condition}

    @property
    def paths(self) -> channel.PathParameters:
        return self.gt.paths

    @property
    def metadata(self) -> Metadata:
        return self.meta


def derive_seeds(seed: int | None, **explicit) -> dict:
    """Independent integer seeds per component from a master seed; explicit ones take precedence."""
    children = np.random.SeedSequence(seed).spawn(len(SEED_COMPONENTS))
    out = {name: int(ch.generate_state(1, dtype=np.uint32)[0]) for name, ch in zip(SEED_COMPONENTS, children)}
    for name in SEED_COMPONENTS:
        if explicit.get(f"seed_{name}") is not None:
            out[name] = int(explicit[f"seed_{name}"])
    return out


class MmWavePositioningSystem:
    def __init__(self, config):
        self.config = config

    def simulate(self, seed: int | None = None, *, snr_db: float | None = None,
                 seed_geometry: int | None = None, seed_gain: int | None = None,
                 seed_beam: int | None = None, seed_noise: int | None = None,
                 injected: dict | None = None, keep_noiseless: bool = False) -> SimulationOutput:
        """Simulate one realisation.

        injected (parity/validation only) may contain: scatterers, ue_position, orientation,
        gain_phases, sigma0_db, beam_phases, symbol_phases, noise_unit.
        """
        cfg = self.config
        inj = injected or {}
        seeds = derive_seeds(seed, seed_geometry=seed_geometry, seed_gain=seed_gain,
                             seed_beam=seed_beam, seed_noise=seed_noise)
        rng = {k: np.random.default_rng(v) for k, v in seeds.items()}
        ph, ar = cfg.physics, cfg.array

        scen = sample_scenario(cfg, rng["geometry"], inj)
        geo = geometry.compute_path_geometry(scen.bs_position, scen.ue_position, scen.orientation,
                                             scen.scatterers, scen.include_los, ph.speed_of_light)
        gain, rho, _ = channel.draw_path_gains(geo, cfg, rng["gain"], phases=inj.get("gain_phases"),
                                               sigma0_db=inj.get("sigma0_db"))
        paths = channel.make_path_parameters(geo, gain, rho, ar.num_tx, ar.num_rx)
        H = channel.build_channel_from_config(paths, cfg)

        F = beamforming.generate_tx_beams(cfg, rng["beam"], phases=inj.get("beam_phases"))
        x = beamforming.generate_symbols(cfg, rng["beam"], phases=inj.get("symbol_phases"))
        clean = observation.noiseless_observation(H, F, x)

        target = cfg.noise.snr_db if snr_db is None else snr_db
        y, nz, n0 = noise.add_awgn_from_target_snr(clean, target, cfg, rng["noise"],
                                                   noise_unit=inj.get("noise_unit"))
        snr_eff = target if cfg.noise.mode == "target_snr" else noise.snr_db_from_n0(clean, n0)

        gt = GroundTruth(position=scen.ue_position, orientation=scen.orientation, condition=scen.condition,
                         paths=paths, scatterers=scen.scatterers, bs_position=scen.bs_position)
        meta = Metadata(snr_db=float(snr_eff), n0=float(n0), seed=seed,
                        seed_geometry=seeds["geometry"], seed_gain=seeds["gain"],
                        seed_beam=seeds["beam"], seed_noise=seeds["noise"], config_name=cfg.name,
                        measured_snr_db=noise.measured_snr_db(clean, nz))
        ctx = ObservationContext(config=cfg, F=F, x=x, n0=float(n0), bs_position=scen.bs_position,
                                 wavelengths=subcarrier_wavelengths(ph.speed_of_light, ph.fc, ph.bandwidth,
                                                                    ph.num_subcarriers,
                                                                    ph.frequency_dependent_array))
        return SimulationOutput(y=y, h=H, gt=gt, meta=meta, context=ctx,
                                noiseless=clean if keep_noiseless else None)


def subsample_beams(sample: SimulationOutput, num_beams: int) -> SimulationOutput:
    """Keep the first ``num_beams`` transmissions (nested beam sets: G=4 is a subset of G=8 ... of G=32).

    Channel, noise realisation and N0 are unchanged, so results at different G are paired; the per-entry SNR of
    Eq. (68) is the same in expectation (fewer beams = less pilot overhead and less total pilot energy)."""
    G = int(num_beams)
    if not 1 <= G <= sample.y.shape[1]:
        raise ValueError(f"num_beams must be in [1, {sample.y.shape[1]}]")
    import dataclasses
    cfg = sample.context.config.replace(**{"beam.num_beams": G})
    ctx = dataclasses.replace(sample.context, config=cfg, F=sample.context.F[:, :G], x=sample.context.x[:, :G])
    return dataclasses.replace(sample, y=sample.y[:, :G], context=ctx,
                               noiseless=None if sample.noiseless is None else sample.noiseless[:, :G])

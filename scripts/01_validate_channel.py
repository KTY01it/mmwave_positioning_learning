"""Level 1 report for channel / observation / SNR (P0 checkpoints). Writes outputs/validation/channel.json
and outputs/validation/beamspace.png (the equivalent of main.m's mesh plot)."""
import json

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _bootstrap import ROOT  # noqa: E402
from positioning.config import load_config  # noqa: E402
from positioning.signal_processing import beamspace as B  # noqa: E402
from positioning.system.simulator import MmWavePositioningSystem  # noqa: E402


def main():
    rep = {}
    # (1) beamspace peak location (Eq. 33), every path, paper config
    cfg = load_config(ROOT / "configs" / "experiments" / "olos.yaml")
    s = MmWavePositioningSystem(cfg).simulate(seed=0)
    Ut, Ur, st, sr = B.dictionaries_from_config(cfg)
    Hc = np.abs(B.beamspace_channel(s.h, Ut, Ur))[0]
    ok = []
    for k in range(s.gt.num_paths):
        it = int(np.argmin(np.abs(st - np.sin(s.gt.paths.aod[k]))))
        ir = int(np.argmin(np.abs(sr - np.sin(s.gt.paths.aoa[k]))))
        win = Hc[max(ir - 1, 0):ir + 2, max(it - 1, 0):it + 2]
        ok.append(bool(Hc[ir, it] == win.max()))
    rep["beamspace_peaks_at_predicted_index"] = ok
    # (2) unitary dictionaries for odd/even N
    rep["unitary_err"] = {n: float(np.max(np.abs(B.dictionary(n, 1.0, 0.5).conj().T @ B.dictionary(n, 1.0, 0.5) - np.eye(n))))
                          for n in (64, 65)}
    # (3) SNR (Eq. 68) within +-0.1 dB, several configs and SNRs
    snr_rep = {}
    for name in ("los", "nlos", "olos"):
        sysm = MmWavePositioningSystem(load_config(ROOT / "configs" / "experiments" / f"{name}.yaml"))
        for snr in (-20.0, 0.0, 10.0):
            sig = noi = 0.0
            for seed in range(100):
                x = sysm.simulate(seed=seed, snr_db=snr, keep_noiseless=True)
                sig += np.sum(np.abs(x.noiseless) ** 2) / x.meta.n0
                noi += np.sum(np.abs(x.y - x.noiseless) ** 2) / x.meta.n0
            snr_rep[f"{name}@{snr:g}"] = float(10 * np.log10(sig / noi))
    rep["measured_snr_db"] = snr_rep
    rep["snr_within_0.1dB"] = all(abs(v - float(k.split("@")[1])) <= 0.1 for k, v in snr_rep.items())
    # (4) determinism
    a = MmWavePositioningSystem(cfg).simulate(seed=123)
    b = MmWavePositioningSystem(cfg).simulate(seed=123)
    rep["bit_exact_same_seed"] = bool(np.array_equal(a.y, b.y))
    print(json.dumps(rep, indent=2))
    out = ROOT / "outputs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "channel.json").write_text(json.dumps(rep, indent=2))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Hc, origin="lower", aspect="auto",
                   extent=[np.degrees(np.arcsin(st[0])), np.degrees(np.arcsin(st[-1])), 0, Hc.shape[0]])
    ax.set_xlabel("AoD grid [deg]"); ax.set_ylabel("AoA grid index"); ax.set_title("|H_check[0]| (OLOS, 3 paths)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "beamspace.png", dpi=150)


if __name__ == "__main__":
    main()

# mmWave single-BS positioning

Python code for single-base-station position and orientation estimation with mmWave MIMO-OFDM.
The physical system and the classical chain (DCS-SOMP, SAGE, EXIP/LMA) follow Shahmansoori et al.,
*Position and Orientation Estimation Through Millimeter-Wave MIMO in 5G Systems*, IEEE TWC 2018.
On top of it, the method selects among competing multipath hypotheses (path addition/removal,
relative pruning, near-endfire AoA flips) with a joint fit / geometry / model-order score.

## Install

```bash
pip install -e .[test]
pip install torch          # learned path and geometry initializers
```

## Layout

| Path | Content |
|---|---|
| `src/positioning/system/` | geometry, arrays, channel, beamforming, noise, simulator |
| `src/positioning/signal_processing/` | beamspace, DCS-SOMP, delay estimation, SAGE |
| `src/positioning/localization/` | LOS / NLOS / OLOS localization, EXIP/LMA |
| `src/positioning/bounds/` | FIM, CRB, PEB/REB |
| `src/positioning/models/` | learned path and geometry initializers |
| `src/positioning/b2f3r*/` | hypothesis generation and selection; `b2f3r_final/` is the final estimator |
| `src/positioning/datasets/`, `evaluation/` | data splits, metrics, benchmark harness |
| `configs/` | system parameters (`shahmansoori_paper.yaml`), scenarios, estimator profiles |
| `scripts/` | validation, dataset generation, training, evaluation |

## Usage

```python
from positioning.config import load_config
from positioning.system.simulator import MmWavePositioningSystem
from positioning.localization.pipeline import ClassicalEstimator
from positioning.bounds.peb_reb import compute_bounds

system = MmWavePositioningSystem(load_config("configs/experiments/nlos.yaml"))
sample = system.simulate(seed=1234, snr_db=-5)
est = ClassicalEstimator()(sample)
bounds = compute_bounds(sample)
```

Training and evaluation:

```bash
python scripts/12_train_path_init.py --samples 400000 --batch 128
python scripts/14_train_geom_init.py --samples 1500000 --epochs 6
python scripts/151_final_eval.py run --tag T --split S --regions iid,lat --G 16,32 --n 900 --arms ...
```

The learned initializers are loaded from `outputs/models/v1r/{path_init,geom_init}/model.pt`.

```bash
python -m pytest -q
```

Tensor convention: `H[N, Nr, Nt]`, `F[N, G, Nt, Mt]`, `y[N, G, Nr]` (subcarrier first).

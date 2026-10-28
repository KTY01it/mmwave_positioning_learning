from pathlib import Path

import numpy as np
import pytest

from positioning.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture
def paper_cfg():
    return load_config(CONFIGS / "shahmansoori_paper.yaml")


@pytest.fixture
def github_cfg():
    return load_config(CONFIGS / "github_reference.yaml")


@pytest.fixture
def small_cfg():
    """Paper physics with small arrays for fast tests (odd Nt -> broadside on grid)."""
    return load_config(CONFIGS / "shahmansoori_paper.yaml",
                       **{"array.num_tx": 17, "array.num_rx": 15, "beam.num_beams": 12,
                          "physics.num_subcarriers": 8})


@pytest.fixture
def rng():
    return np.random.default_rng(0)

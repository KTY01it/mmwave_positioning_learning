import numpy as np
import pytest

from positioning.evaluation import metrics as M


def test_angle_error_wraps():
    assert M.angle_error(np.pi - 0.01, -np.pi + 0.01) == pytest.approx(0.02)


def test_delta_and_threshold():
    assert M.delta_db(2.0, 1.0) == pytest.approx(10 * np.log10(4))
    snr = [-30, -20, -10, 0]
    assert M.snr_threshold(snr, [20, 0.5, 0.2, 0.1]) == -20
    assert M.snr_threshold(snr, [0.1, 5, 0.2, 0.1]) == -10  # must hold for all higher SNRs
    assert M.snr_threshold(snr, [0.1, 0.1, 0.1, 0.1], failure_rates=[0.5, 0, 0, 0]) == -20


def test_outlier_counts_failures():
    assert M.outlier_rate([0.1, np.nan, 10.0], [1, 1, 1]) == pytest.approx(2 / 3)


def test_rmse_and_normalized():
    assert M.rmse([3, 4]) == pytest.approx(np.sqrt(12.5))
    np.testing.assert_allclose(M.normalized_errors([2, 3], [1, 3]), [4, 1])

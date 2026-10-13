import numpy as np
import pytest
from agentcertify.core.risk_control import ConformalRiskController


def test_crc_mathematical_guarantee():
    np.random.seed(42)
    alpha = 0.15
    n_cal = 2000

    scores_cal = np.random.uniform(0.0, 1.0, size=n_cal)
    losses_cal = (scores_cal > 0.5).astype(float)

    controller = ConformalRiskController(alpha=alpha, loss_upper_bound=1.0)
    controller.calibrate(scores_cal, losses_cal)

    scores_test = np.random.uniform(0.0, 1.0, size=5000)
    losses_test = (scores_test > 0.5).astype(float)

    accepted_mask = np.array([controller.is_certified(s) for s in scores_test])

    if np.any(accepted_mask):
        empirical_test_risk = np.mean(losses_test[accepted_mask])
        assert empirical_test_risk <= alpha + 0.05


def test_uncalibrated_evaluation_raises_error():
    controller = ConformalRiskController(alpha=0.05)
    with pytest.raises(RuntimeError):
        controller.is_certified(0.5)


def test_invalid_alpha_parameter():
    with pytest.raises(ValueError):
        ConformalRiskController(alpha=1.5)

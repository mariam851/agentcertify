import pytest
from agentcertify.core.calibration import SplitCalibrationManager


def test_split_holdout_proportions():
    manager = SplitCalibrationManager(seed=42)
    scores = [0.1 * i for i in range(10)]
    losses = [0.0 if i < 5 else 1.0 for i in range(10)]

    cal_set, val_set = manager.split_holdout(scores, losses, cal_ratio=0.6)

    assert cal_set.size == 6
    assert val_set.size == 4
    assert len(cal_set.nonconformity_scores) == 6


def test_split_holdout_invalid_ratio():
    manager = SplitCalibrationManager(seed=42)
    with pytest.raises(ValueError):
        manager.split_holdout([0.1], [0.0], cal_ratio=1.5)


def test_split_holdout_mismatched_lengths():
    manager = SplitCalibrationManager(seed=42)
    with pytest.raises(ValueError):
        manager.split_holdout([0.1, 0.2], [0.0], cal_ratio=0.5)

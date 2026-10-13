import numpy as np
from typing import Sequence, Tuple, Optional
from pydantic import BaseModel


class CalibrationDataset(BaseModel):
    nonconformity_scores: list[float]
    losses: list[float]

    @property
    def size(self) -> int:
        return len(self.nonconformity_scores)


class SplitCalibrationManager:
    """
    Manages hold-out split calibration partitions and cross-validation
    folds for conformal risk certification.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def split_holdout(
        self, scores: Sequence[float], losses: Sequence[float], cal_ratio: float = 0.5
    ) -> Tuple[CalibrationDataset, CalibrationDataset]:
        """
        Splits data into proper calibration and validation sets.
        """
        if not (0.0 < cal_ratio < 1.0):
            raise ValueError(f"cal_ratio must be between 0 and 1, got {cal_ratio}")
        if len(scores) != len(losses):
            raise ValueError("Scores and losses lengths must match.")

        n = len(scores)
        indices = np.arange(n)
        self.rng.shuffle(indices)

        split_idx = int(n * cal_ratio)
        cal_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        cal_set = CalibrationDataset(
            nonconformity_scores=[float(scores[i]) for i in cal_idx],
            losses=[float(losses[i]) for i in cal_idx],
        )
        val_set = CalibrationDataset(
            nonconformity_scores=[float(scores[i]) for i in val_idx],
            losses=[float(losses[i]) for i in val_idx],
        )
        return cal_set, val_set

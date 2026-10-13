import numpy as np
from typing import Sequence, Optional


class ConformalRiskController:
    """
    Implements Conformal Risk Control (CRC) for auditing and bounding
    arbitrary bounded loss functions in generative models and LLM agents.

    Guarantees: E[Loss | Accepted] <= alpha on unseen test instances under exchangeability.
    """

    def __init__(self, alpha: float = 0.05, loss_upper_bound: float = 1.0):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.loss_upper_bound = float(loss_upper_bound)
        self.calibrated_threshold: Optional[float] = None

    def calibrate(
        self, nonconformity_scores: Sequence[float], losses: Sequence[float]
    ) -> float:
        """
        Calculates the risk-controlling threshold lambda_hat on the calibration dataset.
        """
        scores = np.asarray(nonconformity_scores, dtype=float)
        loss_vals = np.asarray(losses, dtype=float)

        if len(scores) != len(loss_vals):
            raise ValueError(
                "Dimensions of nonconformity scores and losses must match."
            )

        n = len(scores)
        if n == 0:
            raise ValueError("Calibration dataset cannot be empty.")

        unique_scores = np.sort(np.unique(scores))
        selected_threshold = float("-inf")
        threshold_found = False

        for thresh in unique_scores:
            accepted_mask = scores <= thresh
            num_accepted = np.sum(accepted_mask)

            if num_accepted == 0:
                continue

            # Upper bound on conditional risk via finite-sample correction
            empirical_risk = (
                np.sum(loss_vals[accepted_mask]) + self.loss_upper_bound
            ) / (num_accepted + 1)

            if empirical_risk <= self.alpha:
                selected_threshold = float(thresh)
                threshold_found = True
            else:
                break

        if not threshold_found:
            selected_threshold = float(unique_scores[0]) - 1e-5

        self.calibrated_threshold = selected_threshold
        return self.calibrated_threshold

    def is_certified(self, score: float) -> bool:
        """
        Verifies if an incoming query's nonconformity score satisfies the statistical bound.
        """
        if self.calibrated_threshold is None:
            raise RuntimeError("Controller must be calibrated prior to evaluation.")
        return float(score) <= self.calibrated_threshold

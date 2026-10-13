from typing import Sequence, Dict, Any
import numpy as np


class ConformalMetrics:
    """
    Evaluates empirical coverage, conditional risk, and acceptance efficiency.
    """

    @staticmethod
    def compute_audit_summary(
        accepted_mask: Sequence[bool], losses: Sequence[float], target_alpha: float
    ) -> Dict[str, Any]:
        mask = np.asarray(accepted_mask, dtype=bool)
        loss_vals = np.asarray(losses, dtype=float)

        if len(mask) != len(loss_vals):
            raise ValueError("Dimensions of accepted_mask and losses must match.")

        total_samples = len(mask)
        if total_samples == 0:
            raise ValueError("Input sequences cannot be empty.")

        accepted_count = int(np.sum(mask))
        acceptance_rate = float(accepted_count / total_samples)

        if accepted_count > 0:
            empirical_risk = float(np.mean(loss_vals[mask]))
        else:
            empirical_risk = 0.0

        is_certified = empirical_risk <= target_alpha

        return {
            "total_samples": total_samples,
            "accepted_count": accepted_count,
            "acceptance_rate": acceptance_rate,
            "empirical_risk": empirical_risk,
            "target_alpha": float(target_alpha),
            "is_risk_controlled": is_certified,
        }

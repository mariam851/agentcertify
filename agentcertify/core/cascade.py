from typing import List
from pydantic import BaseModel


class StepAuditRecord(BaseModel):
    step_id: int
    agent_name: str
    uncertainty_score: float
    risk_budget_consumed: float


class MultiAgentCascadeAuditor:
    """
    Tracks and accumulates finite-sample error probabilities across sequential
    multi-agent reasoning chains via the union bound.
    """

    def __init__(self, total_risk_budget: float = 0.10):
        if not (0.0 < total_risk_budget < 1.0):
            raise ValueError(
                f"Total risk budget must be in (0, 1), got {total_risk_budget}"
            )
        self.total_risk_budget = float(total_risk_budget)
        self.history: List[StepAuditRecord] = []

    def record_step(
        self, agent_name: str, step_alpha: float, uncertainty_score: float
    ) -> bool:
        """
        Records an execution step and validates whether total accumulated risk
        remains within the global budget.
        """
        current_consumed = sum(step.risk_budget_consumed for step in self.history)
        if current_consumed + step_alpha > self.total_risk_budget:
            return False

        record = StepAuditRecord(
            step_id=len(self.history) + 1,
            agent_name=agent_name,
            uncertainty_score=float(uncertainty_score),
            risk_budget_consumed=float(step_alpha),
        )
        self.history.append(record)
        return True

    def get_remaining_budget(self) -> float:
        consumed = sum(step.risk_budget_consumed for step in self.history)
        return max(0.0, self.total_risk_budget - consumed)

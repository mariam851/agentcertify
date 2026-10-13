from typing import Any, Callable, Optional
from pydantic import BaseModel
from agentcertify.core.risk_control import ConformalRiskController


class ExecutionVerdict(BaseModel):
    is_executed: bool
    status: str
    nonconformity_score: float
    output: Optional[Any] = None


class CertifiedAgentInterceptor:
    """
    Middleware interceptor that verifies model execution confidence against
    finite-sample statistical guarantees before triggering external tools.
    """

    def __init__(self, controller: ConformalRiskController):
        self.controller = controller

    def intercept(
        self, score: float, action_fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ExecutionVerdict:
        """
        Executes action_fn only if score is certified under statistical risk bounds.
        """
        if not self.controller.is_certified(score):
            return ExecutionVerdict(
                is_executed=False,
                status="ABORTED_UNCERTAIN",
                nonconformity_score=float(score),
                output=None,
            )

        result = action_fn(*args, **kwargs)
        return ExecutionVerdict(
            is_executed=True,
            status="CERTIFIED_SUCCESS",
            nonconformity_score=float(score),
            output=result,
        )

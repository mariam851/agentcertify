from enum import Enum
from typing import Any, Callable, Optional
from pydantic import BaseModel


class FallbackPolicy(str, Enum):
    ABORT = "ABORT"
    HUMAN_IN_THE_LOOP = "HUMAN_IN_THE_LOOP"
    DEFAULT_PAYLOAD = "DEFAULT_PAYLOAD"


class FallbackHandler(BaseModel):
    policy: FallbackPolicy = FallbackPolicy.ABORT
    default_response: Optional[Any] = None

    def trigger(
        self, context: str, human_callback: Optional[Callable[[str], Any]] = None
    ) -> Any:
        if self.policy == FallbackPolicy.ABORT:
            return {
                "status": "ABORTED",
                "reason": "Execution rejected due to statistical uncertainty",
                "context": context,
            }
        if self.policy == FallbackPolicy.HUMAN_IN_THE_LOOP:
            if human_callback is None:
                raise ValueError(
                    "human_callback is required when policy is HUMAN_IN_THE_LOOP."
                )
            return human_callback(context)
        return self.default_response

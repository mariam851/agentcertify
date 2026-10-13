from typing import Any, List
from agentcertify.wrappers.base import BaseModelWrapper, ModelOutput


class MockAPIWrapper(BaseModelWrapper):
    """
    Standard API model adapter with simulated log-probabilities for black-box LLM auditing.
    """

    def __init__(
        self,
        default_response: str = "Simulated API response",
        default_logprobs: List[float] = None,
    ):
        self.default_response = default_response
        self.default_logprobs = (
            default_logprobs if default_logprobs is not None else [-0.05, -0.10, -0.08]
        )

    def generate_with_uncertainty(self, prompt: str, **kwargs: Any) -> ModelOutput:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return ModelOutput(
            generated_text=self.default_response,
            logprobs=self.default_logprobs,
            metadata={"provider": "mock_api", "prompt_length": len(prompt)},
        )

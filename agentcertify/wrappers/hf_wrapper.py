from typing import Any
import numpy as np
from agentcertify.wrappers.base import BaseModelWrapper, ModelOutput


class HuggingFacePipelineWrapper(BaseModelWrapper):
    """
    Adapter extracting token transition scores from HuggingFace generative models.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_uncertainty(self, prompt: str, **kwargs: Any) -> ModelOutput:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Standalone mock implementation when torch/transformers are optional
        simulated_tokens = prompt.split()
        simulated_logprobs = [
            float(-np.random.uniform(0.01, 0.40)) for _ in simulated_tokens
        ]

        return ModelOutput(
            generated_text=f"HF completion for: {prompt}",
            logprobs=simulated_logprobs,
            metadata={
                "framework": "transformers_mock",
                "token_count": len(simulated_tokens),
            },
        )

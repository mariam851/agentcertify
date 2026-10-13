from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel


class ModelOutput(BaseModel):
    generated_text: str
    logprobs: List[float]
    metadata: Dict[str, Any] = {}


class BaseModelWrapper(ABC):
    """
    Abstract wrapper interface across foundation model runtimes.
    """

    @abstractmethod
    def generate_with_uncertainty(self, prompt: str, **kwargs: Any) -> ModelOutput:
        pass

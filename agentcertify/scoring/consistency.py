from typing import Sequence
import numpy as np


class SemanticConsistencyScorer:
    """
    Evaluates self-consistency across sampled completions.
    Higher nonconformity score denotes higher divergence/hallucination risk.
    """

    @staticmethod
    def compute_disagreement_score(answers: Sequence[str]) -> float:
        if not answers:
            raise ValueError("Answers list cannot be empty.")
        if len(answers) == 1:
            return 0.0

        normalized = [ans.strip().lower() for ans in answers]
        unique_answers, counts = np.unique(normalized, return_counts=True)
        modal_frequency = np.max(counts) / len(normalized)

        # Disagreement is the complement of modal agreement
        return float(1.0 - modal_frequency)

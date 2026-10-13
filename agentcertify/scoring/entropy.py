import numpy as np
from typing import Sequence


class TokenEntropyScorer:
    """
    Computes sequence-level uncertainty metrics from token probability distributions.
    """

    @staticmethod
    def compute_mean_token_entropy(
        token_probabilities: Sequence[Sequence[float]],
    ) -> float:
        """
        Calculates the mean Shannon entropy across generated tokens.
        """
        if not token_probabilities:
            raise ValueError("Token probability sequence cannot be empty.")

        entropies = []
        for probs in token_probabilities:
            p = np.asarray(probs, dtype=float)
            p = p[p > 0.0]
            if len(p) == 0:
                continue
            entropy = -np.sum(p * np.log2(p))
            entropies.append(entropy)

        if not entropies:
            return 0.0

        return float(np.mean(entropies))

    @staticmethod
    def compute_mean_logprob_score(logprobs: Sequence[float]) -> float:
        """
        Transforms log-probabilities into a nonconformity score.
        Higher output indicates higher predictive uncertainty.
        """
        if not logprobs:
            raise ValueError("Log-probabilities sequence cannot be empty.")

        arr = np.asarray(logprobs, dtype=float)
        return float(-np.mean(arr))

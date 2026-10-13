import pytest
from agentcertify.scoring.entropy import TokenEntropyScorer


def test_mean_logprob_score():
    logprobs = [-0.1, -0.2, -0.3]
    score = TokenEntropyScorer.compute_mean_logprob_score(logprobs)
    assert round(score, 4) == 0.2000


def test_empty_logprobs_raises_error():
    with pytest.raises(ValueError):
        TokenEntropyScorer.compute_mean_logprob_score([])


def test_mean_token_entropy():
    distribution = [[0.5, 0.5], [1.0, 0.0]]
    entropy = TokenEntropyScorer.compute_mean_token_entropy(distribution)
    assert round(entropy, 4) == 0.5000

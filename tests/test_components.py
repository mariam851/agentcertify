import pytest
from agentcertify.core.cascade import MultiAgentCascadeAuditor
from agentcertify.scoring.consistency import SemanticConsistencyScorer
from agentcertify.scoring.verifiers import ExecutionVerifier
from agentcertify.runtime.fallback import FallbackHandler, FallbackPolicy


def test_cascade_auditor_budget():
    auditor = MultiAgentCascadeAuditor(total_risk_budget=0.08)
    assert auditor.record_step("RetrievalAgent", step_alpha=0.03, uncertainty_score=0.1)
    assert auditor.record_step("ReasoningAgent", step_alpha=0.04, uncertainty_score=0.2)
    # Exceeds budget (0.03 + 0.04 + 0.02 = 0.09 > 0.08)
    assert not auditor.record_step(
        "ExecutionAgent", step_alpha=0.02, uncertainty_score=0.5
    )


def test_consistency_scorer():
    answers = ["Berlin", "Berlin", "Paris"]
    score = SemanticConsistencyScorer.compute_disagreement_score(answers)
    assert pytest.approx(score, 0.01) == 0.333


def test_json_verifier():
    valid = '{"name": "query", "id": 101}'
    assert ExecutionVerifier.verify_json_schema(valid, ["name", "id"]) == 0.0
    invalid = '{"name": "query"}'
    assert ExecutionVerifier.verify_json_schema(invalid, ["name", "id"]) == 1.0


def test_fallback_handler():
    handler = FallbackHandler(
        policy=FallbackPolicy.DEFAULT_PAYLOAD, default_response="FALLBACK_VAL"
    )
    assert handler.trigger("context") == "FALLBACK_VAL"

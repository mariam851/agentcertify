import pytest
from agentcertify.reporting.metrics import ConformalMetrics
from agentcertify.reporting.visualizer import ConformalVisualizer
from agentcertify.wrappers.api_wrapper import MockAPIWrapper
from agentcertify.wrappers.hf_wrapper import HuggingFacePipelineWrapper


def test_conformal_metrics_and_visualizer():
    mask = [True, True, False]
    losses = [0.0, 0.05, 1.0]
    summary = ConformalMetrics.compute_audit_summary(mask, losses, target_alpha=0.10)

    assert summary["accepted_count"] == 2
    assert summary["acceptance_rate"] == pytest.approx(0.666, 0.01)
    assert summary["is_risk_controlled"]

    report_str = ConformalVisualizer.render_ascii_summary(summary)
    assert "AGENTCERTIFY AUDIT REPORT" in report_str
    assert "PASSED" in report_str


def test_metrics_dimension_mismatch():
    with pytest.raises(ValueError):
        ConformalMetrics.compute_audit_summary([True], [0.1, 0.2], target_alpha=0.05)


def test_model_wrappers():
    api_wrap = MockAPIWrapper()
    api_out = api_wrap.generate_with_uncertainty("test prompt")
    assert len(api_out.logprobs) > 0
    assert api_out.metadata["provider"] == "mock_api"

    hf_wrap = HuggingFacePipelineWrapper()
    hf_out = hf_wrap.generate_with_uncertainty("another test prompt")
    assert len(hf_out.logprobs) == 3
    assert hf_out.metadata["framework"] == "transformers_mock"

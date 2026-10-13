from agentcertify.core.risk_control import ConformalRiskController
from agentcertify.runtime.interceptor import CertifiedAgentInterceptor


def test_interceptor_aborts_on_high_uncertainty():
    controller = ConformalRiskController(alpha=0.05)
    controller.calibrated_threshold = 0.40
    interceptor = CertifiedAgentInterceptor(controller=controller)

    executed = False

    def mock_agent_action():
        nonlocal executed
        executed = True
        return "SUCCESS"

    verdict = interceptor.intercept(score=0.85, action_fn=mock_agent_action)

    assert not verdict.is_executed
    assert verdict.status == "ABORTED_UNCERTAIN"
    assert not executed
    assert verdict.output is None


def test_interceptor_executes_on_certified_score():
    controller = ConformalRiskController(alpha=0.05)
    controller.calibrated_threshold = 0.40
    interceptor = CertifiedAgentInterceptor(controller=controller)

    def mock_agent_action():
        return "ACTION_COMPLETED"

    verdict = interceptor.intercept(score=0.20, action_fn=mock_agent_action)

    assert verdict.is_executed
    assert verdict.status == "CERTIFIED_SUCCESS"
    assert verdict.output == "ACTION_COMPLETED"

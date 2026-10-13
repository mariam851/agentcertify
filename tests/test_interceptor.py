from agentcertify.core.risk_control import ConformalRiskController
from agentcertify.runtime.interceptor import CertifiedAgentInterceptor


def test_interceptor_full_lifecycle():
    controller = ConformalRiskController(alpha=0.10)
    controller.calibrated_threshold = 0.50
    interceptor = CertifiedAgentInterceptor(controller=controller)

    call_count = 0

    def sample_tool(param: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result_{param}"

    # Certified query
    verdict_safe = interceptor.intercept(0.35, sample_tool, param="safe")
    assert verdict_safe.is_executed
    assert verdict_safe.status == "CERTIFIED_SUCCESS"
    assert verdict_safe.output == "result_safe"
    assert call_count == 1

    # Rejected query
    verdict_unsafe = interceptor.intercept(0.85, sample_tool, param="risky")
    assert not verdict_unsafe.is_executed
    assert verdict_unsafe.status == "ABORTED_UNCERTAIN"
    assert verdict_unsafe.output is None
    assert call_count == 1

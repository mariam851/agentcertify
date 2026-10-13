from agentcertify.core.risk_control import ConformalRiskController
from agentcertify.runtime.interceptor import CertifiedAgentInterceptor
from agentcertify.reporting.metrics import ConformalMetrics
from agentcertify.reporting.visualizer import ConformalVisualizer


def run_rag_audit():
    # 1. Setup risk controller for a RAG retrieval system (Alpha = 5%)
    controller = ConformalRiskController(alpha=0.05, loss_upper_bound=1.0)

    # 2. Calibration dataset: nonconformity scores and factual discrepancy losses
    cal_scores = [0.15, 0.32, 0.89, 0.22, 0.76, 0.41, 0.19]
    cal_losses = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    controller.calibrate(cal_scores, cal_losses)

    # 3. Interceptor wrapping database tool
    interceptor = CertifiedAgentInterceptor(controller=controller)

    def query_database(sql: str):
        return f"Query executed successfully: {sql}"

    # 4. Incoming agent requests
    test_scores = [0.18, 0.92, 0.25]
    verdicts = []

    for score in test_scores:
        verdict = interceptor.intercept(
            score=score,
            action_fn=query_database,
            sql="SELECT * FROM medical_records WHERE patient_id = 42;",
        )
        verdicts.append(verdict)

    accepted_mask = [v.is_executed for v in verdicts]
    dummy_losses = [0.0, 1.0, 0.0]

    # 5. Compile and render audit metrics
    metrics = ConformalMetrics.compute_audit_summary(
        accepted_mask, dummy_losses, target_alpha=0.05
    )
    report = ConformalVisualizer.render_ascii_summary(metrics)
    print(report)


if __name__ == "__main__":
    run_rag_audit()

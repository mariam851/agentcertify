from agentcertify.core.cascade import MultiAgentCascadeAuditor
from agentcertify.runtime.fallback import FallbackHandler, FallbackPolicy


def run_financial_pipeline():
    # Strict 10% risk budget across the entire multi-agent transaction pipeline
    cascade = MultiAgentCascadeAuditor(total_risk_budget=0.10)
    fallback = FallbackHandler(policy=FallbackPolicy.ABORT)

    pipeline_steps = [
        {"agent": "DocumentParserAgent", "alpha": 0.03, "uncertainty": 0.12},
        {"agent": "TransactionValidatorAgent", "alpha": 0.04, "uncertainty": 0.18},
        {
            "agent": "WireTransferExecutorAgent",
            "alpha": 0.05,
            "uncertainty": 0.65,
        },  # Exceeds remaining budget
    ]

    for step in pipeline_steps:
        can_proceed = cascade.record_step(
            agent_name=step["agent"],
            step_alpha=step["alpha"],
            uncertainty_score=step["uncertainty"],
        )

        if not can_proceed:
            abort_signal = fallback.trigger(
                f"Risk budget exceeded at step: {step['agent']}"
            )
            print(f"[BLOCKED] Multi-agent execution aborted: {abort_signal}")
            break
        else:
            print(
                f"[PASSED] Step '{step['agent']}' certified. Remaining budget: {cascade.get_remaining_budget():.4f}"
            )


if __name__ == "__main__":
    run_financial_pipeline()

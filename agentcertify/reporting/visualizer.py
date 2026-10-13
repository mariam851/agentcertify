from typing import Dict, Any


class ConformalVisualizer:
    """
    Generates structured ASCII tables and distribution summaries for terminal logging.
    """

    @staticmethod
    def render_ascii_summary(audit_summary: Dict[str, Any]) -> str:
        lines = [
            "=" * 50,
            "           AGENTCERTIFY AUDIT REPORT              ",
            "=" * 50,
            f"Total Queries Evaluated : {audit_summary.get('total_samples', 0)}",
            f"Accepted (Certified)    : {audit_summary.get('accepted_count', 0)}",
            f"Acceptance Rate         : {audit_summary.get('acceptance_rate', 0.0):.2%}",
            f"Empirical Risk Observed : {audit_summary.get('empirical_risk', 0.0):.4f}",
            f"Risk Budget (Alpha)     : {audit_summary.get('target_alpha', 0.0):.4f}",
            f"Statistical Guarantee   : {'PASSED' if audit_summary.get('is_risk_controlled') else 'VIOLATED'}",
            "=" * 50,
        ]
        return "\n".join(lines)

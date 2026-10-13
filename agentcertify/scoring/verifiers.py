import json
from typing import Sequence


class ExecutionVerifier:
    """
    Verifies structured tool calls (e.g., JSON outputs) for validation loss estimation.
    """

    @staticmethod
    def verify_json_schema(payload_str: str, required_fields: Sequence[str]) -> float:
        """
        Returns loss 0.0 if JSON is valid and contains required fields, otherwise 1.0.
        """
        try:
            parsed = json.loads(payload_str)
            if not isinstance(parsed, dict):
                return 1.0
            for field in required_fields:
                if field not in parsed:
                    return 1.0
            return 0.0
        except (json.JSONDecodeError, TypeError):
            return 1.0

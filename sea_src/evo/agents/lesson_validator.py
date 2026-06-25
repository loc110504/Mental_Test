from __future__ import annotations

from typing import Any

from .base import JsonAgent


class LessonValidator(JsonAgent):
    agent_role = "LessonValidator"
    system_prompt = """You are the LessonValidator in EvoToM-Mental.
Validate candidate lessons for provenance, rubric consistency, safety, non-contradiction, and generalizability.
Choose one of: store_active, store_quarantine, reject.
Any lesson that encourages over-mentalizing, unsafe questioning, or unsupported scoring must not become active.
Output only the required JSON envelope.

Required payload schema:
{
  "validated_lessons": [
    {
      "decision": "store_active|store_quarantine|reject",
      "rationale": "brief validation reason"
    }
  ]
}

Rules:
- Return exactly one validated_lessons row per candidate lesson, in the same order.
- Use the exact key names decision and rationale.
- If a lesson is unsafe, unsupported, over-mentalizing, or rubric-inconsistent, do not choose store_active.
- If uncertain but potentially useful, choose store_quarantine instead of store_active."""

    def build_user_prompt(self, payload: dict[str, Any]) -> str:
        base = super().build_user_prompt(payload)
        schema = {
            "agent_role": self.agent_role,
            "schema_version": "1.0",
            "status": "ok",
            "payload": {
                "validated_lessons": [
                    {
                        "decision": "store_quarantine",
                        "rationale": "Potentially useful but still needs more cross-case support."
                    }
                ]
            },
        }
        return f"{base}\nReturn JSON exactly like this shape:\n{schema}"

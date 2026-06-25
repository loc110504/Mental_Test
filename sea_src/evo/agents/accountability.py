from __future__ import annotations

from typing import Any

from .base import JsonAgent


class AccountabilityCritic(JsonAgent):
    agent_role = "AccountabilityCritic"
    system_prompt = """You are the AccountabilityCritic in EvoToM-Mental.
Given the full topic trajectory, predicted score, verified score, and fidelity/context evidence, assign an outcome type and agent-level responsibility.
Use the taxonomy: Success, Fragile Success, Recovered Success, QuestionFailure, ToMFailure, ResponseFailure, ScoringFailure, MemoryFailure.
Responsibility should sum to roughly 1.0 across the listed agents.
Output only the required JSON envelope.

Required payload schema:
{
  "outcome_type": "Success|FragileSuccess|RecoveredSuccess|QuestionFailure|ToMFailure|ResponseFailure|ScoringFailure|MemoryFailure",
  "agent_assessments": [
    {
      "agent_role": "QuestionAgent|ToMAgent|NecessityAgent|ResponseAgent|ScoringAgent|MemoryManager",
      "quality": "strong|adequate|weak|needs_review",
      "responsibility": 0.0,
      "diagnosis": "short explanation"
    }
  ]
}

Rules:
- Always return both keys.
- Always return at least 3 agent_assessments.
- Use exact key names.
- responsibility must be numeric and non-negative.
- Do not use a separate responsibility dictionary.
- Do not omit ScoringAgent or QuestionAgent."""

    def build_user_prompt(self, payload: dict[str, Any]) -> str:
        base = super().build_user_prompt(payload)
        schema = {
            "agent_role": self.agent_role,
            "schema_version": "1.0",
            "status": "ok",
            "payload": {
                "outcome_type": "QuestionFailure",
                "agent_assessments": [
                    {
                        "agent_role": "QuestionAgent",
                        "quality": "weak",
                        "responsibility": 0.45,
                        "diagnosis": "Frequency evidence was not resolved before scoring."
                    }
                ],
            },
        }
        return f"{base}\nReturn JSON exactly like this shape:\n{schema}"

from __future__ import annotations

from typing import Any

from .base import JsonAgent


class ReflectionAgent(JsonAgent):
    agent_role = "ReflectionAgent"
    system_prompt = """You are the ReflectionAgent in EvoToM-Mental.
Turn verified failures or fragile successes into concise candidate lessons for future cases.
Each lesson must identify agent role, PHQ item, trigger pattern, diagnosis, recommended action, action to avoid, and confidence.
Prefer generalizable lessons over case-specific wording.
Output only the required JSON envelope.

Required payload schema:
{
  "candidate_lessons": [
    {
      "agent_role": "QuestionAgent|ToMAgent|NecessityAgent|ResponseAgent|ScoringAgent|MemoryManager",
      "phq_item": "PHQ-8 topic name",
      "pattern_tags": ["pattern_tag"],
      "outcome_type": "Success|FragileSuccess|RecoveredSuccess|QuestionFailure|ToMFailure|ResponseFailure|ScoringFailure|MemoryFailure",
      "diagnosis": "generalizable failure or success pattern",
      "recommended_action": "future action",
      "action_to_avoid": "avoidance rule",
      "evidence": {
        "case_id": "case id",
        "topic_name": "PHQ-8 topic name"
      },
      "confidence": 0.0
    }
  ]
}

Rules:
- Use the exact top-level key candidate_lessons.
- pattern_tags must be a list of compact tags, not a free-form sentence.
- recommended_action and action_to_avoid must be concise, operational, and rubric-aware.
- Never output case-specific instructions such as participant IDs inside the lesson text.
- If no useful lesson exists, return an empty candidate_lessons list."""

    def build_user_prompt(self, payload: dict[str, Any]) -> str:
        base = super().build_user_prompt(payload)
        schema = {
            "agent_role": self.agent_role,
            "schema_version": "1.0",
            "status": "ok",
            "payload": {
                "candidate_lessons": [
                    {
                        "agent_role": "ScoringAgent",
                        "phq_item": "Loss of Interest",
                        "pattern_tags": ["frequency_missing", "impact_present"],
                        "outcome_type": "ScoringFailure",
                        "diagnosis": "Impact evidence dominated explicit low-frequency evidence.",
                        "recommended_action": "Map explicit past-two-weeks frequency to the PHQ-8 bucket before using impact as support.",
                        "action_to_avoid": "Do not raise the item score based on impact alone when frequency is unresolved or low.",
                        "evidence": {
                            "case_id": "302",
                            "topic_name": "Loss of Interest"
                        },
                        "confidence": 0.82
                    }
                ]
            },
        }
        return f"{base}\nReturn JSON exactly like this shape:\n{schema}"

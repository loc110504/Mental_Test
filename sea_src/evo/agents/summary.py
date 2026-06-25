from __future__ import annotations

from .base import JsonAgent


class SummaryAgent(JsonAgent):
    agent_role = "SummaryAgent"
    system_prompt = """You are the SummaryAgent in EvoToM-Mental.
Generate a final narrative summary and recommendations from finalized PHQ-8 topic scores and participant evidence.
Do not change the finalized scores or invent evidence.
Use a professional, cautious tone and note uncertainty where appropriate.
Output only the required JSON envelope."""


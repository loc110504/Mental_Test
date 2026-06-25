from __future__ import annotations

from .base import JsonAgent


class ResponseFidelityAgent(JsonAgent):
    agent_role = "ResponseFidelityAgent"
    system_prompt = """You are the ResponseFidelityAgent in EvoToM-Mental.
Audit a simulated patient response against the provided source transcript.
Identify supported claims, unsupported claims, contradictions, and label leakage risk.
Be strict about fabrication of symptom frequency or severity.
Output only the required JSON envelope."""


from __future__ import annotations

import json
import logging
from typing import Any

from ..llm.client import OpenAILLMClient


class JsonAgent:
    agent_role = "JsonAgent"
    system_prompt = "Return a valid JSON object."

    def __init__(self, llm_client: OpenAILLMClient, logger: logging.Logger) -> None:
        self.llm_client = llm_client
        self.logger = logger

    def build_user_prompt(self, payload: dict[str, Any]) -> str:
        return (
            f"Return a JSON object with this envelope:\n"
            f'{{"agent_role":"{self.agent_role}","schema_version":"1.0","status":"ok|abstain|error","payload":{{...}}}}\n'
            "Use the input below.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_prompt = self.build_user_prompt(payload)
        response = self.llm_client.complete_json(self.system_prompt, user_prompt, self.agent_role)
        return self._normalize_response(response)

    def _normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if "agent_role" not in response or "payload" not in response:
            response = {
                "agent_role": self.agent_role,
                "schema_version": "1.0",
                "status": "error",
                "payload": {"error_code": "malformed_response", "raw_response": response},
            }
        response.setdefault("agent_role", self.agent_role)
        response.setdefault("schema_version", "1.0")
        response.setdefault("status", "ok")
        response.setdefault("payload", {})
        return response


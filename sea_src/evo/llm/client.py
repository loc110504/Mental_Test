from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI

from ..core.schemas import RunConfig
from ..core.serialization import extract_json_object


class OpenAILLMClient:
    def __init__(self, config: RunConfig, logger: logging.Logger) -> None:
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for EvoToM-Mental.")
        kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url
        self.client = OpenAI(**kwargs)
        self.config = config
        self.logger = logger

    def complete_json(self, system_prompt: str, user_prompt: str, agent_role: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.config.llm_max_retries + 1):
            self.logger.info("Calling OpenAI for %s (attempt %s/%s)", agent_role, attempt, self.config.llm_max_retries)
            try:
                response = self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_completion_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                last_error = exc
                self.logger.warning("JSON-mode call failed for %s on attempt %s: %s", agent_role, attempt, exc)
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_completion_tokens,
                    )
                except Exception as fallback_exc:
                    last_error = fallback_exc
                    self.logger.warning("Fallback call failed for %s on attempt %s: %s", agent_role, attempt, fallback_exc)
                    if attempt < self.config.llm_max_retries:
                        time.sleep(self.config.llm_retry_backoff_seconds * attempt)
                    continue

            content = response.choices[0].message.content or "{}"
            self.logger.info("Raw %s response: %s", agent_role, content)
            try:
                return extract_json_object(content)
            except Exception as exc:
                last_error = exc
                self.logger.warning("JSON parse failed for %s on attempt %s: %s", agent_role, attempt, exc)
                repair_messages = messages + [
                    {
                        "role": "assistant",
                        "content": content,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Return only a valid JSON object that follows the required envelope. "
                            "Do not add markdown or commentary."
                        ),
                    },
                ]
                try:
                    repair_response = self.client.chat.completions.create(
                        model=self.config.openai_model,
                        messages=repair_messages,
                        temperature=0,
                        max_tokens=self.config.max_completion_tokens,
                        response_format={"type": "json_object"},
                    )
                    repaired_content = repair_response.choices[0].message.content or "{}"
                    self.logger.info("Repaired %s response: %s", agent_role, repaired_content)
                    return extract_json_object(repaired_content)
                except Exception as repair_exc:
                    last_error = repair_exc
                    self.logger.warning("Repair call failed for %s on attempt %s: %s", agent_role, attempt, repair_exc)
                    if attempt < self.config.llm_max_retries:
                        time.sleep(self.config.llm_retry_backoff_seconds * attempt)
                    continue

        self.logger.error("All LLM attempts failed for %s: %s", agent_role, last_error)
        return {
            "agent_role": agent_role,
            "schema_version": "1.0",
            "status": "error",
            "payload": {
                "error_code": "llm_json_failure",
                "error_message": str(last_error) if last_error else "Unknown LLM failure",
            },
        }

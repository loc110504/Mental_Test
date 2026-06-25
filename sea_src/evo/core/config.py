from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .schemas import RunConfig


def load_config(
    repo_root: str | None = None,
    mode: str = "offline_train",
    interactive: bool = False,
    allow_experience_updates: bool | None = None,
) -> RunConfig:
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    load_dotenv(root / ".env")
    artifacts_root = os.getenv("EVOTOM_ARTIFACTS_ROOT", str(root / "artifacts" / "evo"))
    experience_root = os.getenv("EVOTOM_EXPERIENCE_ROOT", str(Path(artifacts_root) / "experiences"))
    updates = allow_experience_updates
    if updates is None:
        updates = mode in {"offline_train", "dev_eval", "verified_update"}

    config = RunConfig(
        repo_root=str(root),
        mode=mode,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        max_completion_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1200")),
        max_turns_per_topic=int(os.getenv("EVOTOM_MAX_TURNS", "4")),
        question_candidates=int(os.getenv("EVOTOM_QUESTION_CANDIDATES", "4")),
        experience_top_k=int(os.getenv("EVOTOM_EXPERIENCE_TOP_K", "5")),
        llm_max_retries=int(os.getenv("EVOTOM_LLM_MAX_RETRIES", "3")),
        llm_retry_backoff_seconds=float(os.getenv("EVOTOM_LLM_RETRY_BACKOFF", "1.5")),
        artifacts_root=artifacts_root,
        experience_root=experience_root,
        interactive=interactive,
        allow_experience_updates=updates,
    )
    Path(config.artifacts_root).mkdir(parents=True, exist_ok=True)
    Path(config.experience_root).mkdir(parents=True, exist_ok=True)
    return config

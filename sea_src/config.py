import os
import threading
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_PROVIDER = "ollama"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen3:30b"

TEMPERATURE = float(os.getenv("AGENTMENTAL_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("AGENTMENTAL_MAX_TOKENS", "2048"))
CACHE_SEED = None

_rate_limit_lock = threading.Lock()
_rate_limit_windows = {}


def load_environment():
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")


def _env(name, default=None):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _get_provider_config():
    load_environment()

    provider = _env("LLM_PROVIDER", DEFAULT_PROVIDER).lower()

    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": _env("OPENAI_API_KEY"),
            "model_name": _env("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            "base_url": _env("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            "api_type": "openai",
            "max_requests_per_minute": None,
        }

    if provider == "ollama":
        return {
            "provider": "ollama",
            "api_key": _env("OLLAMA_API_KEY", "ollama"),
            "model_name": _env("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            "base_url": _env("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            "api_type": "openai",
            "max_requests_per_minute": None,
        }

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_openai_settings():
    provider_config = _get_provider_config()
    return {
        "model": provider_config["model_name"],
        "api_key": provider_config["api_key"],
        "base_url": provider_config["base_url"],
    }


def get_api_runtime_config():
    provider_config = _get_provider_config()
    return {
        "provider": provider_config["provider"],
        "base_url": provider_config["base_url"],
        "api_key": provider_config["api_key"],
        "model_name": provider_config["model_name"],
        "max_requests_per_minute": provider_config["max_requests_per_minute"],
    }


def get_llm_config():
    provider_config = _get_provider_config()
    config_list = [{
        "model": provider_config["model_name"],
        "api_key": provider_config["api_key"],
        "base_url": provider_config["base_url"],
        "api_type": provider_config["api_type"],
        "tags": [provider_config["provider"], provider_config["model_name"]],
        "price": [0, 0],
    }]

    llm_config = {
        "config_list": config_list,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if CACHE_SEED is not None:
        llm_config["cache_seed"] = CACHE_SEED

    return llm_config


def wait_for_rate_limit():
    provider_config = _get_provider_config()
    max_requests_per_minute = provider_config.get("max_requests_per_minute")
    if not max_requests_per_minute:
        return

    provider_key = provider_config["provider"]
    window_seconds = 60.0

    while True:
        with _rate_limit_lock:
            request_times = _rate_limit_windows.setdefault(provider_key, deque())
            now = time.monotonic()

            while request_times and now - request_times[0] >= window_seconds:
                request_times.popleft()

            if len(request_times) < max_requests_per_minute:
                request_times.append(now)
                return

            sleep_seconds = window_seconds - (now - request_times[0])

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

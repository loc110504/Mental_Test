import autogen
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_api_runtime_config():
    return {
        "base_url": os.getenv("API_BASE_URL", "http://localhost:11434/v1"),
        "api_key": os.getenv("API_KEY", "ollama"),
        "model_name": os.getenv("API_MODEL", "qwen3:30b"),
    }


def get_llm_config():
    model_name = os.getenv("MODEL_NAME", "qwen3:30b")
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        file_location=".",
        filter_dict={"model": [model_name]}
    )

    llm_config = {
        "config_list": config_list,
        "cache_seed": 42,
        "temperature": 0,
        "max_tokens": 2048
    }
    
    return llm_config

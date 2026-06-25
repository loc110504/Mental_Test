import autogen
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_llm_config():
    model_name = os.getenv("MODEL_NAME", "qwen2.5:latest")
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

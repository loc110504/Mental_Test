from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_root: str, run_id: str) -> logging.Logger:
    logger = logging.getLogger(f"evo.{run_id}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir = Path(log_root) / "runs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


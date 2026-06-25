from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..core.config import load_config
from ..core.logging_setup import setup_logging
from ..core.schemas import CaseRecord, GroundTruth, ParticipantProfile, TranscriptTurn
from ..core.serialization import write_json
from ..evaluation.metrics import compute_prediction_metrics, ground_truth_to_dataframe, results_to_dataframe
from ..knowledge.phq8 import PHQ8_TOPIC_TO_FIELD, phq8_topics
from ..llm.client import OpenAILLMClient
from ..pipelines.assessment import AssessmentPipeline


def load_case_from_json(path: str, mode: str, repo_root: str) -> CaseRecord:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    transcript = [
        TranscriptTurn(turn_id=f"tr_{index:03d}", role_name=row.get("roleName", "Unknown"), content=row.get("content", ""))
        for index, row in enumerate(raw.get("real_interview", []), start=1)
    ]
    item_scores = {}
    phq_scores = raw.get("phq8_scores", {})
    items = phq_scores.get("items", {})
    for topic, field_name in PHQ8_TOPIC_TO_FIELD.items():
        if field_name in items:
            item_scores[topic] = int(items[field_name])

    ground_truth = GroundTruth(
        item_scores=item_scores,
        total_score=phq_scores.get("PHQ8_Score"),
        binary_label=phq_scores.get("PHQ8_Binary"),
    )
    participant_profile = ParticipantProfile(source="dataset_json")
    return CaseRecord(
        schema_version="1.0",
        case_id=str(raw.get("Participant_ID", Path(path).stem)),
        mode=mode,
        scale_name="PHQ-8",
        participant_profile=participant_profile,
        transcript=transcript,
        topic_order=phq8_topics(repo_root),
        ground_truth=ground_truth,
        metadata={"source_path": str(Path(path).resolve())},
    )


def run_case_file(case_path: str, mode: str = "offline_train", interactive: bool = False):
    repo_root = str(Path(__file__).resolve().parents[3])
    config = load_config(repo_root=repo_root, mode=mode, interactive=interactive)
    logger = setup_logging(config.artifacts_root, config.run_id)
    llm_client = OpenAILLMClient(config, logger)
    pipeline = AssessmentPipeline(config, llm_client, logger)
    case_record = load_case_from_json(case_path, mode, repo_root)
    logger.info("Running EvoToM-Mental for case %s", case_record.case_id)
    return pipeline.run_case(case_record)


def run_case_directory(case_dir: str, mode: str = "offline_train") -> pd.DataFrame:
    return run_case_directory_filtered(case_dir=case_dir, mode=mode, sample_id=None, limit=None)


def run_case_directory_filtered(
    case_dir: str,
    mode: str = "offline_train",
    sample_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    repo_root = str(Path(__file__).resolve().parents[3])
    config = load_config(repo_root=repo_root, mode=mode, interactive=False)
    logger = setup_logging(config.artifacts_root, config.run_id)
    llm_client = OpenAILLMClient(config, logger)
    pipeline = AssessmentPipeline(config, llm_client, logger)
    results = []
    case_records = []
    paths = resolve_case_paths(case_dir, sample_id=sample_id, limit=limit)
    if sample_id is not None:
        logger.info("Filtered to sample id %s", sample_id)
    if limit is not None:
        logger.info("Limiting run to first %s case files", limit)

    for path in paths:
        case_record = load_case_from_json(str(path), mode, repo_root)
        case_records.append(case_record)
        logger.info("Running EvoToM-Mental for case %s", case_record.case_id)
        result = pipeline.run_case(case_record)
        results.append(result.to_dict())
    df = results_to_dataframe(results)
    run_dir = Path(config.artifacts_root) / "runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "predictions.csv", index=False)
    write_json(run_dir / "results.json", results)

    if case_records:
        gold_df = ground_truth_to_dataframe(case_records)
        gold_df.to_csv(run_dir / "gold.csv", index=False)
        metrics = compute_prediction_metrics(gold_df, df)
        write_json(run_dir / "metrics.json", metrics)
    return df


def resolve_case_paths(case_dir: str, sample_id: str | None = None, limit: int | None = None) -> list[Path]:
    paths = sorted(Path(case_dir).glob("*.json"))
    if sample_id is not None:
        target_name = f"{sample_id}.json"
        paths = [path for path in paths if path.name == target_name]
        if not paths:
            raise FileNotFoundError(f"Sample '{sample_id}' was not found in {case_dir}.")
    if limit is not None:
        paths = paths[:limit]
    return paths

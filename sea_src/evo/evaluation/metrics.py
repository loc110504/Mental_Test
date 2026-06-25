from __future__ import annotations

import pandas as pd

from ..knowledge.phq8 import PHQ8_TOPIC_TO_FIELD


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for row in results:
        item_scores = row.get("topic_scores", {})
        base = {
            "identifier": row["case_id"],
            "total": row["total_score"],
            "classes": row["binary_label"],
        }
        for index, (_, value) in enumerate(item_scores.items(), start=1):
            base[f"item{index}"] = value if value is not None else 0
        rows.append(base)
    return pd.DataFrame(rows)


def ground_truth_to_dataframe(case_records: list) -> pd.DataFrame:
    rows = []
    topic_order = list(PHQ8_TOPIC_TO_FIELD.keys())
    for case in case_records:
        base = {
            "identifier": case.case_id,
            "total": case.ground_truth.total_score,
            "classes": case.ground_truth.binary_label,
        }
        for index, topic in enumerate(topic_order, start=1):
            base[f"item{index}"] = case.ground_truth.item_scores.get(topic, 0)
        rows.append(base)
    return pd.DataFrame(rows)


def compute_prediction_metrics(df_gold: pd.DataFrame, df_pred: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

    common_ids = set(df_gold["identifier"]).intersection(df_pred["identifier"])
    gold = df_gold[df_gold["identifier"].isin(common_ids)].sort_values("identifier").reset_index(drop=True)
    pred = df_pred[df_pred["identifier"].isin(common_ids)].sort_values("identifier").reset_index(drop=True)
    return {
        "mae": mean_absolute_error(gold["total"], pred["total"]),
        "accuracy": accuracy_score(gold["classes"], pred["classes"]),
        "macro_f1": f1_score(gold["classes"], pred["classes"], average="macro", zero_division=0),
        "f1_control": f1_score(gold["classes"], pred["classes"], labels=[0], pos_label=0, zero_division=0),
        "f1_depressed": f1_score(gold["classes"], pred["classes"], labels=[1], pos_label=1, zero_division=0),
        "kappa": cohen_kappa_score(gold["classes"], pred["classes"]),
        "num_cases": int(len(common_ids)),
    }

from __future__ import annotations

import json
from pathlib import Path


PHQ8_TOPIC_TO_FIELD = {
    "Loss of Interest": "PHQ8_NoInterest",
    "Depressed Mood": "PHQ8_Depressed",
    "Sleep Problems": "PHQ8_Sleep",
    "Fatigue or Low Energy": "PHQ8_Tired",
    "Appetite or Weight Changes": "PHQ8_Appetite",
    "Low Self-Worth": "PHQ8_Failure",
    "Concentration Difficulties": "PHQ8_Concentrating",
    "Psychomotor Changes": "PHQ8_Moving",
}


def load_question_templates(repo_root: str) -> dict[str, list[str]]:
    path = Path(repo_root) / "scales" / "PHQ-8.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scoring_standards(repo_root: str) -> dict[str, dict[str, str]]:
    path = Path(repo_root) / "scales" / "scoring_standards.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["PHQ-8"]


def phq8_topics(repo_root: str) -> list[str]:
    return list(load_question_templates(repo_root).keys())


def build_static_knowledge(repo_root: str) -> dict[str, object]:
    question_templates = load_question_templates(repo_root)
    standards = load_scoring_standards(repo_root)
    items: dict[str, dict[str, object]] = {}
    for topic, rubric in standards.items():
        items[topic] = {
            "rubric": rubric,
            "seed_questions": question_templates.get(topic, []),
            "time_window": "past two weeks",
            "scoring_priority": ["presence", "time_frame", "frequency", "impact_support"],
            "questioning_rules": [
                "Ask neutral follow-up questions.",
                "Do not use ToM hypotheses as facts.",
                "Resolve frequency before letting impact dominate scoring.",
            ],
        }
    return {
        "scale_name": "PHQ-8",
        "time_window": "past_two_weeks",
        "items": items,
    }


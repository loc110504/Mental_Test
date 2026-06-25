from __future__ import annotations

import re
from typing import Any


def _score_from_day_count(day_count: int) -> int:
    if day_count <= 1:
        return 0
    if day_count <= 6:
        return 1
    if day_count <= 11:
        return 2
    return 3


def extract_frequency_signal(text: str) -> tuple[int | None, str | None, float]:
    lowered = text.lower()

    exact_patterns = [
        (r"\b(0|1)\s*day\b", 0),
        (r"\b([2-6])\s*days\b", 1),
        (r"\b([7-9]|10|11)\s*days\b", 2),
        (r"\b(12|13|14)\s*days\b", 3),
    ]
    for pattern, score in exact_patterns:
        match = re.search(pattern, lowered)
        if match:
            return score, match.group(0), 0.95

    range_match = re.search(r"\b(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\s*days?\b", lowered)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        midpoint = round((start + end) / 2)
        return _score_from_day_count(midpoint), range_match.group(0), 0.9

    pair_match = re.search(r"\b(\d{1,2})\s*(?:or|and)\s*(\d{1,2})\s*days?\b", lowered)
    if pair_match:
        first = int(pair_match.group(1))
        second = int(pair_match.group(2))
        midpoint = round((first + second) / 2)
        return _score_from_day_count(midpoint), pair_match.group(0), 0.88

    phrase_map = {
        0: ["not at all", "never", "no days", "none in the last two weeks"],
        1: ["several days", "a few days", "few days", "sometimes", "occasionally", "couple of days"],
        2: ["more than half the days", "most days", "over half the days"],
        3: ["nearly every day", "almost every day", "every day", "daily", "all the time"],
    }
    for score, phrases in phrase_map.items():
        for phrase in phrases:
            if phrase in lowered:
                confidence = 0.9 if "day" in phrase else 0.72
                return score, phrase, confidence

    return None, None, 0.0


def has_time_anchor(turns) -> bool:
    for turn in turns:
        question = turn.question_text.lower()
        answer = turn.answer_text.lower()
        if "past two weeks" in question or "last two weeks" in question:
            return True
        if "past two weeks" in answer or "last two weeks" in answer or "14 days" in answer:
            return True
    return False


def detect_symptom_acknowledgement(topic_state) -> bool:
    return any(statement.extracted_evidence.symptom for statement in topic_state.statements)


def build_guardrail_summary(topic_state, fidelity_records: list[dict[str, Any]]) -> dict[str, Any]:
    best_signal: tuple[int | None, str | None, float] = (None, None, 0.0)
    conflicting_scores: set[int] = set()
    for statement in topic_state.statements:
        signal = extract_frequency_signal(statement.raw_text)
        if signal[0] is not None:
            conflicting_scores.add(signal[0])
        if signal[2] > best_signal[2]:
            best_signal = signal

    time_known = has_time_anchor(topic_state.turns)
    symptom_present = detect_symptom_acknowledgement(topic_state)
    mean_fidelity = 1.0
    if fidelity_records:
        scores = [float(row.get("payload", {}).get("fidelity_score", 1.0)) for row in fidelity_records]
        mean_fidelity = sum(scores) / len(scores)

    score, frequency_text, confidence = best_signal
    conflicting = len(conflicting_scores) > 1
    can_score = (
        score is not None
        and time_known
        and confidence >= 0.72
        and not conflicting
        and mean_fidelity >= 0.45
    )
    must_resolve: list[str] = []
    if score is None:
        must_resolve.append("frequency")
    if not time_known:
        must_resolve.append("time")
    if conflicting:
        must_resolve.append("frequency_conflict")
    if mean_fidelity < 0.45:
        must_resolve.append("response_fidelity")

    return {
        "symptom_present": symptom_present,
        "time_known": time_known,
        "frequency_score": score,
        "frequency_text": frequency_text,
        "frequency_confidence": confidence,
        "frequency_conflicting": conflicting,
        "mean_fidelity": mean_fidelity,
        "can_score_confidently": can_score,
        "must_resolve": must_resolve,
        "should_abstain_if_no_more_turns": not can_score,
    }


def finalize_scoring_payload(
    topic_name: str,
    payload: dict[str, Any],
    guardrail: dict[str, Any],
    used_lesson_ids: list[str],
) -> dict[str, Any]:
    result = dict(payload)
    result.setdefault("evidence_spans", {})
    result.setdefault("rubric_checks", {})
    result["rubric_checks"]["time_window_satisfied"] = guardrail["time_known"]
    result["rubric_checks"]["frequency_resolved"] = guardrail["frequency_score"] is not None
    result["rubric_checks"]["fidelity_safe"] = guardrail["mean_fidelity"] >= 0.45
    result["rubric_checks"]["guardrail_can_score"] = guardrail["can_score_confidently"]
    result.setdefault("used_lesson_ids", used_lesson_ids)

    if not guardrail["can_score_confidently"]:
        result["predicted_score"] = None
        result["abstained"] = True
        result["abstain_reason"] = ",".join(guardrail["must_resolve"]) or "insufficient_rubric_evidence"
        result["uncertainty"] = max(float(result.get("uncertainty", 0.7)), 0.8)
        summary = result.get("reasoning_summary", "").strip()
        addon = (
            "Rule-based PHQ-8 guardrail blocked confident scoring because "
            + ", ".join(guardrail["must_resolve"])
            + "."
        )
        result["reasoning_summary"] = f"{summary} {addon}".strip()
        return result

    if guardrail["frequency_score"] is not None:
        llm_score = result.get("predicted_score")
        rule_score = int(guardrail["frequency_score"])
        if llm_score is None or llm_score != rule_score:
            result["predicted_score"] = rule_score
            result["rubric_checks"]["rule_based_override"] = True
            evidence = guardrail["frequency_text"]
            if evidence:
                result["evidence_spans"].setdefault("frequency", [])
                if evidence not in result["evidence_spans"]["frequency"]:
                    result["evidence_spans"]["frequency"].append(evidence)
            summary = result.get("reasoning_summary", "").strip()
            addon = f"Rule-based PHQ-8 override applied from explicit frequency evidence: {guardrail['frequency_text']}."
            result["reasoning_summary"] = f"{summary} {addon}".strip()
        else:
            result["rubric_checks"]["rule_based_override"] = False

    return result

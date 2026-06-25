from __future__ import annotations

from typing import Any

from ..core.schemas import VerificationResult


class SelfEvolutionPipeline:
    def __init__(self, accountability_critic, reflection_agent, lesson_validator, experience_tree, memory_manager, logger) -> None:
        self.accountability_critic = accountability_critic
        self.reflection_agent = reflection_agent
        self.lesson_validator = lesson_validator
        self.experience_tree = experience_tree
        self.memory_manager = memory_manager
        self.logger = logger

    def run(
        self,
        case_record,
        topic_state,
        topic_score_payload: dict[str, Any],
        scoring_rubric: dict[str, str],
        fidelity_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        gold = case_record.ground_truth.item_scores.get(topic_state.topic_name)
        pred = topic_score_payload.get("predicted_score")
        if gold is None:
            return None

        absolute_error = abs((pred or 0) - gold)
        verification = VerificationResult(
            ground_truth_score=gold,
            predicted_score=pred,
            absolute_error=absolute_error,
            normalized_reward=1 - absolute_error / 3.0,
        )

        accountability_input = {
            "case_id": case_record.case_id,
            "topic_name": topic_state.topic_name,
            "topic_state": topic_state.to_dict(),
            "predicted_score": pred,
            "ground_truth_score": gold,
            "absolute_error": absolute_error,
            "fidelity_records": fidelity_records,
            "rubric": scoring_rubric,
        }
        accountability = self.accountability_critic.run(accountability_input)

        reflection_input = {
            "case_id": case_record.case_id,
            "topic_name": topic_state.topic_name,
            "accountability_report": accountability,
            "topic_state": topic_state.to_dict(),
            "rubric": scoring_rubric,
        }
        reflection = self.reflection_agent.run(reflection_input)

        candidate_lessons = reflection.get("payload", {}).get("candidate_lessons", [])
        for lesson in candidate_lessons:
            lesson.setdefault("evidence", {})
            lesson["evidence"].setdefault("case_id", case_record.case_id)
            lesson["evidence"].setdefault("topic_name", topic_state.topic_name)
            lesson["evidence"].setdefault(
                "supporting_span_ids",
                [statement.node_id for statement in topic_state.statements],
            )
            lesson["confidence"] = float(lesson.get("confidence", max(0.4, verification.normalized_reward)))

        validation = self.lesson_validator.run(
            {
                "case_id": case_record.case_id,
                "topic_name": topic_state.topic_name,
                "candidate_lessons": candidate_lessons,
                "rubric": scoring_rubric,
                "topic_state": topic_state.to_dict(),
            }
        )
        validated_rows = validation.get("payload", {}).get("validated_lessons", [])
        normalized_rows = []
        for idx, row in enumerate(validated_rows):
            lesson = candidate_lessons[idx] if idx < len(candidate_lessons) else {}
            normalized_rows.append(
                {
                    "decision": row.get("decision", "reject"),
                    "rationale": row.get("rationale", ""),
                    "lesson": lesson,
                }
            )
        stored_lessons = self.experience_tree.store_lessons(
            [row for row in normalized_rows if row["decision"] in {"store_active", "store_quarantine"}]
        )
        used_lesson_ids: list[str] = []
        for lesson_ids in topic_state.retrieved_lessons.values():
            used_lesson_ids.extend(lesson_ids)
        self.memory_manager.record_retrieval_feedback(
            sorted(set(used_lesson_ids)),
            verification.normalized_reward,
        )
        maintenance = self.memory_manager.maintain()
        return {
            "verification": verification.to_dict(),
            "accountability": accountability,
            "reflection": reflection,
            "validation": validation,
            "stored_lessons": [lesson.to_dict() for lesson in stored_lessons],
            "memory_maintenance": maintenance,
        }

import json
import logging
from pathlib import Path

from evo.agents.accountability import AccountabilityCritic
from evo.agents.lesson_validator import LessonValidator
from evo.agents.reflection import ReflectionAgent
from evo.agents.response_fidelity import ResponseFidelityAgent
from evo.core.config import load_config as load_evo_config
from evo.core.serialization import append_jsonl, write_json
from evo.knowledge.phq8 import PHQ8_TOPIC_TO_FIELD
from evo.llm.client import OpenAILLMClient
from evo.memory.experience_tree import ExperienceTree
from evo.memory.manager import MemoryManager
from evo.memory.participant_tree import extract_evidence


class SelfEvolutionManager:
    ALLOWED_OUTCOME_TYPES = {
        "Success",
        "FragileSuccess",
        "RecoveredSuccess",
        "QuestionFailure",
        "ToMFailure",
        "ResponseFailure",
        "ScoringFailure",
        "MemoryFailure",
    }
    ALLOWED_AGENT_ROLES = {
        "QuestionAgent",
        "ToMAgent",
        "NecessityAgent",
        "ResponseAgent",
        "ScoringAgent",
        "MemoryManager",
    }
    ALLOWED_QUALITIES = {"strong", "adequate", "weak", "needs_review"}
    ALLOWED_VALIDATION_DECISIONS = {"store_active", "store_quarantine", "reject"}

    def __init__(self, logger, automated=False, repo_root=None):
        self.logger = logger
        self.config = load_evo_config(
            repo_root=repo_root,
            mode="offline_train" if automated else "inference",
            interactive=not automated,
            allow_experience_updates=automated,
        )
        self.llm_client = OpenAILLMClient(self.config, logger)
        self.experience_tree = ExperienceTree(self.config.experience_root)
        self.memory_manager = MemoryManager(self.experience_tree, logger)
        self.accountability_critic = AccountabilityCritic(self.llm_client, logger)
        self.reflection_agent = ReflectionAgent(self.llm_client, logger)
        self.lesson_validator = LessonValidator(self.llm_client, logger)
        self.response_fidelity_agent = ResponseFidelityAgent(self.llm_client, logger)

    def get_ground_truth_score(self, topic_name, scale_scores):
        item_scores = scale_scores.get("items", {})
        field_name = PHQ8_TOPIC_TO_FIELD.get(topic_name)
        if not field_name:
            return None
        score = item_scores.get(field_name)
        return score if isinstance(score, int) else None

    def case_dir(self, case_id):
        path = Path(self.config.artifacts_root) / "cases" / str(case_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def derive_pattern_tags(self, topic_name, memory_context):
        tags = set()
        short_term_memory = memory_context.get("short_term_memory", [])
        tom_hypotheses = memory_context.get("tom_hypotheses", [])
        evidence_rows = [
            extract_evidence(item.get("raw_text") or item.get("content", ""))
            for item in short_term_memory
        ]

        if any(
            row.symptom or row.emotion or row.self_belief or row.social_belief
            for row in evidence_rows
        ):
            tags.add("explicit_symptom_present")

        if any(row.frequency for row in evidence_rows):
            tags.add("frequency_present")
        else:
            tags.add("frequency_missing")

        if any(row.time_frame for row in evidence_rows):
            tags.add("time_present")
        else:
            tags.add("time_missing")

        if any(row.impact for row in evidence_rows):
            tags.add("impact_present")
        else:
            tags.add("impact_missing")

        if any(row.social_belief for row in evidence_rows):
            tags.add("social_belief_signal")
        if any(row.self_belief for row in evidence_rows):
            tags.add("negative_self_belief_signal")
        if any("possible_minimization" in row.inconsistency_flags for row in evidence_rows):
            tags.add("possible_minimization")
        if any(row.frequency and row.impact for row in evidence_rows):
            tags.add("frequency_and_impact_present")
        if any(
            (
                "several_days" in row.frequency
                or "rarely" in row.frequency
            ) and row.impact
            for row in evidence_rows
        ):
            tags.add("impact_high_frequency_low_candidate")
        if any(hyp.get("label") == "possible_symptom_minimization" for hyp in tom_hypotheses):
            tags.add("possible_minimization")
        if any(hyp.get("verification_status") == "unverified" for hyp in tom_hypotheses):
            tags.add("tom_unverified")
        if not short_term_memory:
            tags.add("topic_cold_start")
        return sorted(tags)

    def retrieve_lessons(self, case_id, topic_name, agent_role, pattern_tags, top_k=None):
        top_k = top_k or self.config.experience_top_k
        retrievals = self.experience_tree.retrieve_with_scores(
            agent_role,
            topic_name,
            pattern_tags,
            top_k,
        )
        lesson_dicts = [item["lesson"].to_dict() for item in retrievals]
        lesson_ids = [item["lesson"].lesson_id for item in retrievals]
        self.memory_manager.register_retrieval(lesson_ids)
        append_jsonl(
            self.case_dir(case_id) / "agent_traces" / f"{topic_name}_retrievals.jsonl",
            {
                "agent_role": agent_role,
                "pattern_tags": pattern_tags,
                "retrievals": [
                    {
                        "lesson": item["lesson"].to_dict(),
                        "score": item["score"],
                        "breakdown": item["breakdown"],
                    }
                    for item in retrievals
                ],
            },
        )
        return lesson_dicts, lesson_ids

    def audit_response_fidelity(
        self,
        case_id,
        topic_name,
        question_text,
        answer_text,
        current_topic_history,
        real_interview,
    ):
        if self.config.mode == "inference":
            return None

        payload = {
            "topic_name": topic_name,
            "question_text": question_text,
            "simulated_answer": answer_text,
            "topic_history": current_topic_history,
            "source_transcript": real_interview,
            "instructions": {
                "focus": [
                    "supported claims",
                    "unsupported claims",
                    "contradictions",
                    "label leakage risk",
                ]
            },
        }
        response = self.response_fidelity_agent.run(payload)
        normalized = self._normalize_fidelity(response, answer_text)
        append_jsonl(
            self.case_dir(case_id) / "agent_traces" / f"{topic_name}_fidelity.jsonl",
            normalized,
        )
        return normalized

    def run_topic_self_evolution(
        self,
        case_id,
        topic_name,
        predicted_score,
        scoring_payload,
        scoring_rubric,
        topic_history,
        memory_context,
        retrieved_lesson_ids,
        pattern_tags,
        ground_truth_score,
        fidelity_records,
        necessity_records,
    ):
        if ground_truth_score is None or not self.config.allow_experience_updates:
            return None

        case_dir = self.case_dir(case_id)
        absolute_error = abs((predicted_score or 0) - ground_truth_score)
        verification = {
            "ground_truth_score": ground_truth_score,
            "predicted_score": predicted_score,
            "absolute_error": absolute_error,
            "normalized_reward": 1 - absolute_error / 3.0,
        }

        topic_state = {
            "topic_name": topic_name,
            "topic_history": topic_history,
            "memory_context": memory_context,
            "retrieved_lessons": retrieved_lesson_ids,
            "pattern_tags": pattern_tags,
            "scoring_payload": scoring_payload,
            "fidelity_records": fidelity_records,
            "necessity_records": necessity_records,
        }

        accountability_raw = self.accountability_critic.run(
            {
                "case_id": case_id,
                "topic_name": topic_name,
                "topic_state": topic_state,
                "predicted_score": predicted_score,
                "ground_truth_score": ground_truth_score,
                "absolute_error": absolute_error,
                "fidelity_records": fidelity_records,
                "rubric": scoring_rubric,
                "taxonomy": [
                    "Success",
                    "FragileSuccess",
                    "RecoveredSuccess",
                    "QuestionFailure",
                    "ToMFailure",
                    "ResponseFailure",
                    "ScoringFailure",
                    "MemoryFailure",
                ],
            }
        )
        accountability = self._normalize_accountability(
            accountability_raw,
            topic_history,
            memory_context,
            predicted_score,
            ground_truth_score,
            fidelity_records,
            necessity_records,
        )

        reflection_raw = self.reflection_agent.run(
            {
                "case_id": case_id,
                "topic_name": topic_name,
                "accountability_report": accountability,
                "topic_state": topic_state,
                "rubric": scoring_rubric,
                "lesson_schema": {
                    "agent_role": "QuestionAgent|ToMAgent|NecessityAgent|ResponseAgent|ScoringAgent|MemoryManager",
                    "phq_item": topic_name,
                    "pattern_tags": pattern_tags,
                    "outcome_type": accountability["payload"]["outcome_type"],
                    "diagnosis": "brief diagnosis",
                    "recommended_action": "future action",
                    "action_to_avoid": "avoidance rule",
                    "evidence": {
                        "case_id": case_id,
                        "topic_name": topic_name,
                    },
                    "confidence": "0.0-1.0",
                },
            }
        )
        candidate_lessons = self._normalize_candidate_lessons(
            reflection_raw,
            accountability,
            topic_name,
            case_id,
            pattern_tags,
            topic_history,
            scoring_payload,
            ground_truth_score,
        )

        validation_raw = self.lesson_validator.run(
            {
                "case_id": case_id,
                "topic_name": topic_name,
                "candidate_lessons": candidate_lessons,
                "rubric": scoring_rubric,
                "topic_state": topic_state,
                "validation_policy": {
                    "check_provenance": True,
                    "check_rubric_consistency": True,
                    "check_safety": True,
                    "check_generalizability": True,
                    "allow_active_only_if_safe_and_supported": True,
                },
            }
        )
        validated_rows = self._normalize_validated_lessons(validation_raw, candidate_lessons)
        stored_lessons = self.experience_tree.store_lessons(
            [row for row in validated_rows if row["decision"] in {"store_active", "store_quarantine"}]
        )
        used_lesson_ids = sorted(
            {
                lesson_id
                for lesson_ids in retrieved_lesson_ids.values()
                for lesson_id in lesson_ids
            }
        )
        self.memory_manager.record_retrieval_feedback(
            used_lesson_ids,
            verification["normalized_reward"],
        )
        maintenance = self.memory_manager.maintain()
        result = {
            "verification": verification,
            "accountability": accountability,
            "reflection": {
                "agent_role": "ReflectionAgent",
                "schema_version": "1.0",
                "status": "ok",
                "payload": {"candidate_lessons": candidate_lessons},
            },
            "validation": {
                "agent_role": "LessonValidator",
                "schema_version": "1.0",
                "status": "ok",
                "payload": {"validated_lessons": validated_rows},
            },
            "stored_lessons": [lesson.to_dict() for lesson in stored_lessons],
            "memory_maintenance": maintenance,
        }
        write_json(case_dir / "verification" / f"{topic_name}.json", result)
        return result

    def _normalize_fidelity(self, response, answer_text):
        payload = response.get("payload", {}) if isinstance(response, dict) else {}
        fidelity_score = payload.get("fidelity_score")
        if not isinstance(fidelity_score, (int, float)):
            lowered = answer_text.lower()
            fidelity_score = 0.6 if "not sure" in lowered else 0.8
        return {
            "agent_role": "ResponseFidelityAgent",
            "schema_version": "1.0",
            "status": response.get("status", "ok") if isinstance(response, dict) else "ok",
            "payload": {
                "fidelity_score": float(fidelity_score),
                "supported_claims": payload.get("supported_claims", []),
                "unsupported_claims": payload.get("unsupported_claims", []),
                "contradictions": payload.get("contradictions", []),
                "label_leakage_risk": payload.get("label_leakage_risk", 0.0),
            },
        }

    def _normalize_accountability(
        self,
        response,
        topic_history,
        memory_context,
        predicted_score,
        ground_truth_score,
        fidelity_records,
        necessity_records,
    ):
        payload = response.get("payload", {}) if isinstance(response, dict) else {}
        outcome_type = payload.get("outcome_type")
        agent_assessments = payload.get("agent_assessments")
        if (
            outcome_type in self.ALLOWED_OUTCOME_TYPES
            and isinstance(agent_assessments, list)
        ):
            cleaned = self._clean_agent_assessments(agent_assessments, outcome_type)
            if cleaned:
                return {
                    "agent_role": "AccountabilityCritic",
                    "schema_version": "1.0",
                    "status": response.get("status", "ok") if isinstance(response, dict) else "ok",
                    "payload": {
                        "outcome_type": outcome_type,
                        "agent_assessments": cleaned,
                    },
                }

        absolute_error = abs((predicted_score or 0) - ground_truth_score)
        statements = memory_context.get("short_term_memory", [])
        evidence_rows = [
            extract_evidence(item.get("raw_text") or item.get("content", ""))
            for item in statements
        ]
        has_frequency = any(row.frequency for row in evidence_rows)
        has_time = any(row.time_frame for row in evidence_rows)
        has_unverified_tom = any(
            hyp.get("verification_status") == "unverified"
            for hyp in memory_context.get("tom_hypotheses", [])
        )
        low_fidelity = any(
            record.get("payload", {}).get("fidelity_score", 1.0) < 0.5
            for record in fidelity_records
        )
        stopped_early = bool(necessity_records) and necessity_records[-1].get("necessity_score", 0) == 0 and len(topic_history) <= 1

        if absolute_error == 0:
            if not has_frequency or not has_time:
                outcome_type = "FragileSuccess"
            elif len(topic_history) > 1:
                outcome_type = "RecoveredSuccess"
            else:
                outcome_type = "Success"
        elif low_fidelity:
            outcome_type = "ResponseFailure"
        elif not has_frequency or not has_time or stopped_early:
            outcome_type = "QuestionFailure"
        elif has_unverified_tom:
            outcome_type = "ToMFailure"
        else:
            outcome_type = "ScoringFailure"

        responsibility = {
            "QuestionAgent": 0.15,
            "ToMAgent": 0.1,
            "NecessityAgent": 0.1,
            "ResponseAgent": 0.05,
            "ScoringAgent": 0.5,
            "MemoryManager": 0.1,
        }
        if outcome_type == "QuestionFailure":
            responsibility.update({"QuestionAgent": 0.45, "NecessityAgent": 0.25, "ScoringAgent": 0.15, "ToMAgent": 0.1, "MemoryManager": 0.05})
        elif outcome_type == "ToMFailure":
            responsibility.update({"ToMAgent": 0.45, "QuestionAgent": 0.25, "ScoringAgent": 0.15, "MemoryManager": 0.1, "NecessityAgent": 0.05})
        elif outcome_type == "ResponseFailure":
            responsibility.update({"ResponseAgent": 0.55, "ScoringAgent": 0.2, "QuestionAgent": 0.1, "ToMAgent": 0.05, "NecessityAgent": 0.05, "MemoryManager": 0.05})
        elif outcome_type == "ScoringFailure":
            responsibility.update({"ScoringAgent": 0.65, "QuestionAgent": 0.15, "ToMAgent": 0.05, "NecessityAgent": 0.05, "MemoryManager": 0.1})
        elif outcome_type == "FragileSuccess":
            responsibility.update({"ScoringAgent": 0.35, "QuestionAgent": 0.25, "NecessityAgent": 0.15, "ToMAgent": 0.15, "MemoryManager": 0.1})
        elif outcome_type == "RecoveredSuccess":
            responsibility.update({"QuestionAgent": 0.25, "ToMAgent": 0.2, "NecessityAgent": 0.15, "ScoringAgent": 0.25, "MemoryManager": 0.15})

        agent_assessments = self._clean_agent_assessments([
            {
                "agent_role": agent_role,
                "quality": "strong" if share <= 0.15 and outcome_type in {"Success", "RecoveredSuccess"} else "needs_review",
                "responsibility": share,
                "diagnosis": f"{agent_role} contribution to {outcome_type}",
            }
            for agent_role, share in responsibility.items()
        ], outcome_type)
        return {
            "agent_role": "AccountabilityCritic",
            "schema_version": "1.0",
            "status": "ok",
            "payload": {
                "outcome_type": outcome_type,
                "agent_assessments": agent_assessments,
            },
        }

    def _normalize_candidate_lessons(
        self,
        response,
        accountability,
        topic_name,
        case_id,
        pattern_tags,
        topic_history,
        scoring_payload,
        ground_truth_score,
    ):
        payload = response.get("payload", {}) if isinstance(response, dict) else {}
        lessons = payload.get("candidate_lessons") or payload.get("lessons") or []
        normalized = []
        for lesson in lessons:
            normalized_lesson = self._normalize_single_lesson(
                lesson,
                topic_name,
                case_id,
                pattern_tags,
                accountability["payload"]["outcome_type"],
            )
            if normalized_lesson:
                normalized.append(normalized_lesson)
        if normalized:
            return normalized

        outcome_type = accountability["payload"]["outcome_type"]
        predicted = scoring_payload.get("score", 0)
        summary = scoring_payload.get("summary", "")
        fallback = []
        if outcome_type in {"QuestionFailure", "FragileSuccess"}:
            fallback.append(
                self._lesson_template(
                    "QuestionAgent",
                    topic_name,
                    pattern_tags,
                    outcome_type,
                    "Frequency or time-window evidence was not resolved safely before scoring.",
                    "Ask directly about how many days in the past two weeks the symptom occurred before relying on impact.",
                    "Do not let functional impact dominate scoring when explicit PHQ-8 frequency is still unclear.",
                    case_id,
                )
            )
            fallback.append(
                self._lesson_template(
                    "NecessityAgent",
                    topic_name,
                    pattern_tags,
                    outcome_type,
                    "The interview loop stopped before the rubric-critical evidence was complete.",
                    "Continue follow-up when frequency or past-two-weeks anchoring remains unresolved.",
                    "Do not stop early just because the participant gave some impact detail.",
                    case_id,
                )
            )
        if outcome_type in {"ToMFailure", "FragileSuccess"}:
            fallback.append(
                self._lesson_template(
                    "ToMAgent",
                    topic_name,
                    pattern_tags,
                    outcome_type,
                    "A ToM signal needed neutral verification rather than assumption.",
                    "Frame latent beliefs or minimization as hypotheses with alternative explanations and missing verification.",
                    "Do not treat a ToM hypothesis as a confirmed symptom or scoring fact.",
                    case_id,
                )
            )
        if outcome_type in {"ScoringFailure", "FragileSuccess", "RecoveredSuccess"}:
            fallback.append(
                self._lesson_template(
                    "ScoringAgent",
                    topic_name,
                    pattern_tags,
                    outcome_type,
                    f"Predicted score {predicted} diverged from verified label {ground_truth_score} or required caution.",
                    "State the explicit frequency evidence and map it to the PHQ-8 bucket before finalizing the score.",
                    "Do not override explicit low frequency with impact alone.",
                    case_id,
                )
            )
        if outcome_type == "ResponseFailure":
            fallback.append(
                self._lesson_template(
                    "ResponseAgent",
                    topic_name,
                    pattern_tags,
                    outcome_type,
                    "The simulated response likely introduced unsupported detail.",
                    "If the transcript does not support a frequency claim, answer with uncertainty instead of inventing a PHQ-8 bucket.",
                    "Do not fabricate nearly-every-day or similar frequency expressions without transcript support.",
                    case_id,
                )
            )
        if not fallback:
            fallback.append(
                self._lesson_template(
                    "MemoryManager",
                    topic_name,
                    pattern_tags,
                    outcome_type,
                    f"Case pattern required better memory support. Summary: {summary[:120]}",
                    "Retrieve only role- and topic-matched lessons with strong pattern overlap.",
                    "Do not surface weak or contradictory lessons as if they were authoritative.",
                    case_id,
                )
            )
        return fallback

    def _normalize_single_lesson(self, lesson, topic_name, case_id, pattern_tags, outcome_type):
        if not isinstance(lesson, dict):
            return None
        agent_role = lesson.get("agent_role") or lesson.get("AgentRole") or "ScoringAgent"
        if agent_role not in self.ALLOWED_AGENT_ROLES:
            return None
        phq_item = lesson.get("phq_item") or lesson.get("PHQ_item") or topic_name
        lesson_outcome = lesson.get("outcome_type") or outcome_type
        if lesson_outcome not in self.ALLOWED_OUTCOME_TYPES:
            lesson_outcome = outcome_type
        diagnosis = str(lesson.get("diagnosis", "")).strip()
        recommended_action = str(lesson.get("recommended_action", "")).strip()
        action_to_avoid = str(lesson.get("action_to_avoid", "")).strip()
        if not diagnosis or not recommended_action or not action_to_avoid:
            return None
        normalized_tags = self._normalize_pattern_tags(
            lesson.get("pattern_tags") or self._coerce_trigger_pattern(lesson.get("trigger_pattern"), pattern_tags)
        )
        if not normalized_tags:
            normalized_tags = self._normalize_pattern_tags(pattern_tags)
        return {
            "agent_role": agent_role,
            "phq_item": phq_item,
            "pattern_tags": normalized_tags,
            "outcome_type": lesson_outcome,
            "diagnosis": diagnosis,
            "recommended_action": recommended_action,
            "action_to_avoid": action_to_avoid,
            "evidence": lesson.get("evidence") or {
                "case_id": case_id,
                "topic_name": topic_name,
            },
            "confidence": self._coerce_confidence(lesson.get("confidence", 0.6)),
        }

    def _normalize_validated_lessons(self, response, candidate_lessons):
        payload = response.get("payload", {}) if isinstance(response, dict) else {}
        rows = payload.get("validated_lessons", [])
        normalized = []
        for idx, lesson in enumerate(candidate_lessons):
            row = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
            decision = row.get("decision")
            rationale = row.get("rationale", "")
            if decision not in self.ALLOWED_VALIDATION_DECISIONS:
                decision, rationale = self._fallback_validation_decision(lesson)
            normalized.append(
                {
                    "decision": decision,
                    "rationale": str(rationale).strip() or "Validator output was incomplete; fallback decision applied.",
                    "lesson": lesson,
                }
            )
        return normalized

    def _fallback_validation_decision(self, lesson):
        diagnosis = lesson.get("diagnosis", "").lower()
        recommended_action = lesson.get("recommended_action", "").lower()
        action_to_avoid = lesson.get("action_to_avoid", "").lower()
        unsafe = any(
            term in recommended_action
            for term in ["assume the participant", "treat as confirmed", "lead the participant"]
        )
        contradictory = "override explicit frequency" in recommended_action and "do not override explicit low frequency" in action_to_avoid
        if unsafe or contradictory:
            return "reject", "Unsafe or contradictory lesson."
        confidence = float(lesson.get("confidence", 0.0))
        generalizable = len(lesson.get("pattern_tags", [])) > 0 and "participant " not in diagnosis
        if confidence >= 0.75 and generalizable:
            return "store_active", "Supported, rubric-consistent, and generalizable."
        if confidence >= 0.5:
            return "store_quarantine", "Potentially useful but needs more support."
        return "reject", "Low-confidence lesson."

    @staticmethod
    def _coerce_trigger_pattern(trigger_pattern, default_tags):
        if isinstance(trigger_pattern, list) and trigger_pattern:
            return trigger_pattern
        if isinstance(trigger_pattern, str) and trigger_pattern.strip():
            return [trigger_pattern.strip().lower().replace(" ", "_")]
        return default_tags

    @staticmethod
    def _lesson_template(
        agent_role,
        topic_name,
        pattern_tags,
        outcome_type,
        diagnosis,
        recommended_action,
        action_to_avoid,
        case_id,
    ):
        return {
            "agent_role": agent_role,
            "phq_item": topic_name,
            "pattern_tags": pattern_tags,
            "outcome_type": outcome_type,
            "diagnosis": diagnosis,
            "recommended_action": recommended_action,
            "action_to_avoid": action_to_avoid,
            "evidence": {
                "case_id": case_id,
                "topic_name": topic_name,
                "supporting_cases": [case_id],
            },
            "confidence": 0.7,
        }

    def _clean_agent_assessments(self, rows, outcome_type):
        cleaned = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            agent_role = row.get("agent_role")
            if agent_role not in self.ALLOWED_AGENT_ROLES:
                continue
            diagnosis = str(row.get("diagnosis", "")).strip()
            if not diagnosis:
                diagnosis = f"{agent_role} contribution to {outcome_type}"
            quality = row.get("quality", "needs_review")
            if quality not in self.ALLOWED_QUALITIES:
                quality = "needs_review"
            try:
                responsibility = float(row.get("responsibility", 0.0))
            except (TypeError, ValueError):
                responsibility = 0.0
            responsibility = max(0.0, responsibility)
            cleaned.append(
                {
                    "agent_role": agent_role,
                    "quality": quality,
                    "responsibility": responsibility,
                    "diagnosis": diagnosis,
                }
            )

        seen = {item["agent_role"] for item in cleaned}
        for required_role in ["QuestionAgent", "ScoringAgent"]:
            if required_role not in seen:
                cleaned.append(
                    {
                        "agent_role": required_role,
                        "quality": "needs_review",
                        "responsibility": 0.0,
                        "diagnosis": f"{required_role} contribution to {outcome_type}",
                    }
                )
        if len(cleaned) < 3:
            for extra_role in ["ToMAgent", "NecessityAgent", "MemoryManager"]:
                if extra_role not in {item["agent_role"] for item in cleaned}:
                    cleaned.append(
                        {
                            "agent_role": extra_role,
                            "quality": "needs_review",
                            "responsibility": 0.0,
                            "diagnosis": f"{extra_role} contribution to {outcome_type}",
                        }
                    )
                if len(cleaned) >= 3:
                    break

        total = sum(item["responsibility"] for item in cleaned)
        if total > 0:
            for item in cleaned:
                item["responsibility"] = round(item["responsibility"] / total, 4)
        else:
            uniform = round(1.0 / len(cleaned), 4)
            for item in cleaned:
                item["responsibility"] = uniform
        return cleaned

    @staticmethod
    def _normalize_pattern_tags(tags):
        normalized = []
        if not isinstance(tags, list):
            return normalized
        for tag in tags:
            text = str(tag).strip().lower().replace(" ", "_")
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @staticmethod
    def _coerce_confidence(value):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.6
        return min(1.0, max(0.0, confidence))

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agents.accountability import AccountabilityCritic
from ..agents.context_eval import ContextEvaluationAgent
from ..agents.lesson_validator import LessonValidator
from ..agents.question import QuestionAgent
from ..agents.reflection import ReflectionAgent
from ..agents.response_fidelity import ResponseFidelityAgent
from ..agents.scoring import ScoringAgent
from ..agents.summary import SummaryAgent
from ..agents.tom import ToMAgent
from ..core.schemas import CaseRunResult, Provenance, QuestionTurn, SufficiencyVector, TopicScore
from ..core.serialization import append_jsonl, write_json
from ..knowledge.phq8 import build_static_knowledge
from ..knowledge.phq8_rules import build_guardrail_summary, finalize_scoring_payload
from ..memory.experience_tree import ExperienceTree
from ..memory.manager import MemoryManager
from ..memory.participant_tree import ParticipantEvidenceTree, derive_pattern_tags
from ..pipelines.self_evolution import SelfEvolutionPipeline
from ..simulators.transcript_patient import SimulatedResponseAgent


class AssessmentPipeline:
    def __init__(self, config, llm_client, logger) -> None:
        self.config = config
        self.logger = logger
        self.llm_client = llm_client
        self.static_knowledge = build_static_knowledge(config.repo_root)
        self.experience_tree = ExperienceTree(config.experience_root)
        self.memory_manager = MemoryManager(self.experience_tree, logger)
        self.question_agent = QuestionAgent(llm_client, logger)
        self.tom_agent = ToMAgent(llm_client, logger)
        self.context_agent = ContextEvaluationAgent(llm_client, logger)
        self.scoring_agent = ScoringAgent(llm_client, logger)
        self.response_agent = SimulatedResponseAgent(llm_client, logger)
        self.fidelity_agent = ResponseFidelityAgent(llm_client, logger)
        self.accountability_critic = AccountabilityCritic(llm_client, logger)
        self.reflection_agent = ReflectionAgent(llm_client, logger)
        self.lesson_validator = LessonValidator(llm_client, logger)
        self.summary_agent = SummaryAgent(llm_client, logger)
        self.self_evolution = SelfEvolutionPipeline(
            self.accountability_critic,
            self.reflection_agent,
            self.lesson_validator,
            self.experience_tree,
            self.memory_manager,
            logger,
        )

    def run_case(self, case_record) -> CaseRunResult:
        case_dir = Path(self.config.artifacts_root) / "cases" / case_record.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        participant_tree = ParticipantEvidenceTree(case_record.participant_profile.to_dict())
        topic_scores: dict[str, int | None] = {}
        abstained_topics: list[str] = []
        final_topic_states: dict[str, Any] = {}

        for topic_name in case_record.topic_order:
            topic_state = participant_tree.ensure_topic(topic_name)
            fidelity_records: list[dict[str, Any]] = []
            question_history: list[dict[str, Any]] = []
            while len(topic_state.turns) < self.config.max_turns_per_topic:
                pattern_tags = derive_pattern_tags(topic_state)
                participant_tree.set_pattern_tags(topic_name, pattern_tags)
                guardrail = build_guardrail_summary(topic_state, fidelity_records)
                question_retrieval = self.experience_tree.retrieve_with_scores(
                    "QuestionAgent",
                    topic_name,
                    pattern_tags,
                    self.config.experience_top_k,
                )
                question_lessons = [item["lesson"] for item in question_retrieval]
                topic_state.retrieved_lessons["QuestionAgent"] = [lesson.lesson_id for lesson in question_lessons]
                self.memory_manager.register_retrieval(topic_state.retrieved_lessons["QuestionAgent"])
                question_payload = self.question_agent.run(
                    {
                        "topic_name": topic_name,
                        "topic_history": question_history,
                        "participant_memory_snapshot": participant_tree.topic_snapshot(topic_name),
                        "active_tom_hypotheses": [item.to_dict() for item in topic_state.hypotheses if item.verification_status == "unverified"],
                        "retrieved_lessons": [lesson.to_dict() for lesson in question_lessons],
                        "context_gaps": self._context_gaps(topic_state.sufficiency),
                        "question_budget": {
                            "current_turn": len(topic_state.turns),
                            "max_turns": self.config.max_turns_per_topic,
                        },
                        "phq8_guardrail": guardrail,
                        "seed_questions": self.static_knowledge["items"][topic_name]["seed_questions"],
                    }
                )
                append_jsonl(
                    case_dir / "agent_traces" / f"{topic_name}_retrievals.jsonl",
                    {
                        "agent_role": "QuestionAgent",
                        "pattern_tags": pattern_tags,
                        "retrievals": [self._serialize_retrieval(item) for item in question_retrieval],
                    },
                )
                question_text = question_payload.get("payload", {}).get("question_text") or self._fallback_question(topic_name)
                question_turn = QuestionTurn(
                    turn_index=len(topic_state.turns) + 1,
                    question_text=question_text,
                    question_type=question_payload.get("payload", {}).get("question_type", "followup"),
                    target_gaps=question_payload.get("payload", {}).get("target_gaps", []),
                    verifies_hypothesis_ids=question_payload.get("payload", {}).get("verifies_hypothesis_ids", []),
                    utility_breakdown=question_payload.get("payload", {}).get("utility_breakdown", {}),
                )

                answer_text, provenance, fidelity = self._obtain_response(case_record, topic_name, question_text, question_history)
                question_turn.answer_text = answer_text
                question_turn.answer_provenance = provenance
                question_turn.fidelity_score = fidelity.get("payload", {}).get("fidelity_score") if fidelity else None
                topic_state.turns.append(question_turn)
                question_history.append({"question": question_text, "response": answer_text})
                statement = participant_tree.add_statement(topic_name, "participant", question_turn.turn_index, answer_text, provenance)

                if fidelity:
                    fidelity_records.append(fidelity)
                    append_jsonl(case_dir / "agent_traces" / f"{topic_name}_fidelity.jsonl", fidelity)

                pattern_tags = derive_pattern_tags(topic_state)
                participant_tree.set_pattern_tags(topic_name, pattern_tags)
                tom_retrieval = self.experience_tree.retrieve_with_scores(
                    "ToMAgent",
                    topic_name,
                    pattern_tags,
                    self.config.experience_top_k,
                )
                tom_lessons = [item["lesson"] for item in tom_retrieval]
                topic_state.retrieved_lessons["ToMAgent"] = [lesson.lesson_id for lesson in tom_lessons]
                self.memory_manager.register_retrieval(topic_state.retrieved_lessons["ToMAgent"])
                tom_output = self.tom_agent.run(
                    {
                        "topic_name": topic_name,
                        "recent_statement": statement.to_dict(),
                        "topic_history": question_history,
                        "participant_memory_snapshot": participant_tree.topic_snapshot(topic_name),
                        "retrieved_lessons": [lesson.to_dict() for lesson in tom_lessons],
                    }
                )
                participant_tree.add_hypotheses(topic_name, tom_output.get("payload", {}).get("hypotheses", []))
                participant_tree.set_pattern_tags(topic_name, derive_pattern_tags(topic_state))
                append_jsonl(
                    case_dir / "agent_traces" / f"{topic_name}_retrievals.jsonl",
                    {
                        "agent_role": "ToMAgent",
                        "pattern_tags": pattern_tags,
                        "retrievals": [self._serialize_retrieval(item) for item in tom_retrieval],
                    },
                )

                context_output = self.context_agent.run(
                    {
                        "topic_name": topic_name,
                        "topic_history": question_history,
                        "participant_memory_snapshot": participant_tree.topic_snapshot(topic_name),
                        "rubric": self.static_knowledge["items"][topic_name]["rubric"],
                        "phq8_guardrail": build_guardrail_summary(topic_state, fidelity_records),
                    }
                )
                topic_state.sufficiency = self._build_sufficiency(context_output.get("payload", {}))
                participant_tree.update_sufficiency(topic_name, topic_state.sufficiency)

                append_jsonl(case_dir / "agent_traces" / f"{topic_name}_turns.jsonl", {"question": question_payload, "tom": tom_output, "context": context_output})
                if not self._should_continue(
                    context_output.get("payload", {}),
                    topic_state,
                    fidelity_records,
                    self.config.max_turns_per_topic,
                ):
                    break

            participant_tree.set_pattern_tags(topic_name, derive_pattern_tags(topic_state))
            scoring_guardrail = build_guardrail_summary(topic_state, fidelity_records)
            scoring_retrieval = self.experience_tree.retrieve_with_scores(
                "ScoringAgent",
                topic_name,
                topic_state.pattern_tags,
                self.config.experience_top_k,
            )
            scoring_lessons = [item["lesson"] for item in scoring_retrieval]
            topic_state.retrieved_lessons["ScoringAgent"] = [lesson.lesson_id for lesson in scoring_lessons]
            self.memory_manager.register_retrieval(topic_state.retrieved_lessons["ScoringAgent"])
            scoring_output = self.scoring_agent.run(
                {
                    "topic_name": topic_name,
                    "topic_history": question_history,
                    "participant_memory_snapshot": participant_tree.topic_snapshot(topic_name),
                    "retrieved_lessons": [lesson.to_dict() for lesson in scoring_lessons],
                    "rubric": self.static_knowledge["items"][topic_name]["rubric"],
                    "phq8_guardrail": scoring_guardrail,
                }
            )
            append_jsonl(
                case_dir / "agent_traces" / f"{topic_name}_retrievals.jsonl",
                {
                    "agent_role": "ScoringAgent",
                    "pattern_tags": topic_state.pattern_tags,
                    "retrievals": [self._serialize_retrieval(item) for item in scoring_retrieval],
                },
            )
            finalized_payload = finalize_scoring_payload(
                topic_name,
                scoring_output.get("payload", {}),
                scoring_guardrail,
                topic_state.retrieved_lessons["ScoringAgent"],
            )
            topic_score = self._build_topic_score(topic_name, finalized_payload, topic_state.retrieved_lessons["ScoringAgent"])
            participant_tree.set_score(topic_name, topic_score)
            topic_scores[topic_name] = topic_score.predicted_score
            if topic_score.abstained:
                abstained_topics.append(topic_name)

            if self.config.allow_experience_updates:
                evolution = self.self_evolution.run(
                    case_record,
                    topic_state,
                    finalized_payload,
                    self.static_knowledge["items"][topic_name]["rubric"],
                    fidelity_records,
                )
                if evolution:
                    write_json(case_dir / "verification" / f"{topic_name}.json", evolution)

            final_topic_states[topic_name] = topic_state.to_dict()
            write_json(case_dir / "topics" / f"{topic_name}.json", topic_state.to_dict())

        total_score = sum(score or 0 for score in topic_scores.values())
        binary_label = 1 if total_score >= 10 else 0
        summary_output = self.summary_agent.run(
            {
                "case_id": case_record.case_id,
                "scale_name": case_record.scale_name,
                "topic_scores": topic_scores,
                "participant_tree": participant_tree.export(),
                "abstained_topics": abstained_topics,
            }
        )
        summary_payload = summary_output.get("payload", {})
        result = CaseRunResult(
            case_id=case_record.case_id,
            total_score=summary_payload.get("total_score", total_score),
            binary_label=summary_payload.get("binary_label", binary_label),
            topic_scores=topic_scores,
            abstained_topics=abstained_topics,
            report_summary=summary_payload.get("narrative_summary", ""),
            recommendations=summary_payload.get("recommendations", []),
        )

        write_json(case_dir / "participant_tree.json", participant_tree.export())
        write_json(case_dir / "case_state.json", case_record.to_dict())
        write_json(case_dir / "result.json", result.to_dict())
        return result

    def _obtain_response(self, case_record, topic_name: str, question_text: str, question_history: list[dict[str, str]]):
        if self.config.mode in {"offline_train", "dev_eval", "verified_update"}:
            response = self.response_agent.run(
                {
                    "case_id": case_record.case_id,
                    "topic_name": topic_name,
                    "question_text": question_text,
                    "participant_profile": case_record.participant_profile.to_dict(),
                    "transcript": [turn.to_dict() for turn in case_record.transcript],
                    "topic_history": question_history,
                }
            )
            payload = response.get("payload", {})
            answer_text = payload.get("response_text", "I'm not sure.")
            provenance = Provenance(
                source_type="transcript_simulation",
                source_span_ids=payload.get("used_source_span_ids", []),
            )
            fidelity = self.fidelity_agent.run(
                {
                    "topic_name": topic_name,
                    "question_text": question_text,
                    "response_text": answer_text,
                    "transcript": [turn.to_dict() for turn in case_record.transcript],
                    "topic_history": question_history,
                }
            )
            return answer_text, provenance, fidelity

        if self.config.interactive:
            answer_text = input(f"{question_text}\n> ").strip()
        else:
            answer_text = ""
        return answer_text, Provenance(source_type="real_user", source_span_ids=[]), None

    @staticmethod
    def _build_sufficiency(payload: dict[str, Any]) -> SufficiencyVector:
        sufficiency = payload.get("sufficiency", {})
        return SufficiencyVector(
            presence=float(sufficiency.get("presence", 0.0)),
            time=float(sufficiency.get("time", 0.0)),
            frequency=float(sufficiency.get("frequency", 0.0)),
            impact=float(sufficiency.get("impact", 0.0)),
            consistency=float(sufficiency.get("consistency", 1.0)),
            tom_verification=float(sufficiency.get("tom_verification", 1.0)),
        )

    @staticmethod
    def _context_gaps(sufficiency: SufficiencyVector) -> list[str]:
        gaps = []
        for name, value in sufficiency.to_dict().items():
            if value < 0.7:
                gaps.append(name)
        if not gaps:
            gaps = ["presence", "time", "frequency"]
        return gaps

    @staticmethod
    def _should_continue(payload: dict[str, Any], topic_state, fidelity_records: list[dict[str, Any]], max_turns: int) -> bool:
        if len(topic_state.turns) >= max_turns:
            return False
        guardrail = build_guardrail_summary(topic_state, fidelity_records)
        if guardrail["must_resolve"]:
            return True
        if payload.get("recommended_next_action") == "score_now":
            return False
        necessity = float(payload.get("necessity_score", 0.0))
        must_resolve = payload.get("must_resolve_before_scoring", [])
        return necessity > 0.45 or bool(must_resolve)

    @staticmethod
    def _build_topic_score(topic_name: str, payload: dict[str, Any], used_lesson_ids: list[str]) -> TopicScore:
        return TopicScore(
            node_id=f"score_{topic_name.lower().replace(' ', '_')}",
            node_type="TopicScoreNode",
            topic_name=topic_name,
            predicted_score=payload.get("predicted_score"),
            abstained=bool(payload.get("abstained", False)),
            abstain_reason=payload.get("abstain_reason"),
            uncertainty=float(payload.get("uncertainty", 0.0)),
            frequency_evidence=payload.get("evidence_spans", {}).get("frequency", []),
            impact_evidence=payload.get("evidence_spans", {}).get("impact", []),
            reasoning_summary=payload.get("reasoning_summary", ""),
            used_lesson_ids=used_lesson_ids,
            rubric_checks=payload.get("rubric_checks", {}),
        )

    def _fallback_question(self, topic_name: str) -> str:
        prompts = self.static_knowledge["items"][topic_name]["seed_questions"]
        return prompts[0] if prompts else f"Over the past two weeks, how has {topic_name.lower()} affected you?"

    @staticmethod
    def _serialize_retrieval(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "lesson_id": item["lesson"].lesson_id,
            "score": item["score"],
            "breakdown": item["breakdown"],
            "lesson": item["lesson"].to_dict(),
        }

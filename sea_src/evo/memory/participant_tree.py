from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from typing import Any

from ..core.schemas import Provenance, StatementEvidence, StatementNode, SufficiencyVector, ToMHypothesis, TopicScore, TopicState


def _collect_matches(text: str, mapping: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for label, patterns in mapping.items():
        if any(pattern in lowered for pattern in patterns):
            found.append(label)
    return found


def extract_evidence(text: str) -> StatementEvidence:
    frequency_map = {
        "rarely": ["rarely", "hardly ever", "once or twice"],
        "several_days": ["a few days", "several days", "sometimes", "occasionally"],
        "more_than_half": ["more than half", "most days", "often"],
        "nearly_every_day": ["every day", "nearly every day", "daily", "all the time"],
    }
    time_map = {
        "past_two_weeks": ["past two weeks", "last two weeks", "lately", "recently"],
        "weeks": ["weeks", "week"],
        "months": ["months", "month"],
    }
    impact_map = {
        "work": ["work", "job", "office"],
        "social": ["friends", "family", "social", "people"],
        "sleep_routine": ["sleep", "bed", "night"],
        "activities": ["activity", "activities", "hobby", "things", "do much"],
    }
    emotion_map = {
        "sadness": ["sad", "down", "depressed", "hopeless"],
        "anxiety": ["anxious", "nervous", "worried"],
        "guilt": ["guilty", "failure", "worthless"],
        "numbness": ["numb", "empty"],
    }
    symptom_map = {
        "reduced_interest": ["no interest", "little interest", "don't want to do much", "lost interest", "no pleasure"],
        "depressed_mood": ["depressed", "hopeless", "down"],
        "sleep_problem": ["insomnia", "can't sleep", "sleep too much", "trouble sleeping"],
        "fatigue": ["tired", "exhausted", "low energy", "fatigue"],
        "appetite_change": ["poor appetite", "overeating", "not hungry", "eating too much"],
        "low_self_worth": ["worthless", "failure", "bad about myself"],
        "concentration_problem": ["can't focus", "concentrate", "mind foggy"],
        "psychomotor_change": ["restless", "fidgety", "slowed down"],
    }

    self_belief = []
    social_belief = []
    inconsistency = []
    lowered = text.lower()
    if any(term in lowered for term in ["burden", "bother others", "don't want to bother", "don't want to trouble"]):
        social_belief.append("perceived_burdensomeness")
    if any(term in lowered for term in ["i'm okay though", "probably okay", "not that bad"]):
        inconsistency.append("possible_minimization")
    if any(term in lowered for term in ["useless", "not useful", "failure", "worthless"]):
        self_belief.append("negative_self_belief")

    return StatementEvidence(
        symptom=_collect_matches(text, symptom_map),
        frequency=_collect_matches(text, frequency_map),
        time_frame=_collect_matches(text, time_map),
        impact=_collect_matches(text, impact_map),
        emotion=_collect_matches(text, emotion_map),
        self_belief=self_belief,
        social_belief=social_belief,
        inconsistency_flags=inconsistency,
    )


class ParticipantEvidenceTree:
    def __init__(self, participant_info: dict[str, Any]) -> None:
        self.participant_info = participant_info
        self.topics: dict[str, TopicState] = {}
        self.edges: list[dict[str, str]] = []

    def ensure_topic(self, topic_name: str) -> TopicState:
        if topic_name not in self.topics:
            self.topics[topic_name] = TopicState(topic_name=topic_name)
            self.edges.append({"source": "User", "target": topic_name, "relation": "assessed_on"})
        return self.topics[topic_name]

    def add_statement(
        self,
        topic_name: str,
        speaker: str,
        turn_id: int,
        raw_text: str,
        provenance: Provenance,
    ) -> StatementNode:
        topic = self.ensure_topic(topic_name)
        node = StatementNode(
            node_id=f"stmt_{uuid.uuid4().hex[:12]}",
            node_type="StatementNode",
            topic_name=topic_name,
            speaker=speaker,
            turn_id=turn_id,
            raw_text=raw_text,
            extracted_evidence=extract_evidence(raw_text),
            provenance=provenance,
        )
        topic.statements.append(node)
        self.edges.append({"source": topic_name, "target": node.node_id, "relation": "has_statement"})
        return node

    def add_hypotheses(self, topic_name: str, hypotheses: list[dict[str, Any]]) -> list[ToMHypothesis]:
        topic = self.ensure_topic(topic_name)
        created: list[ToMHypothesis] = []
        for item in hypotheses:
            node = ToMHypothesis(
                node_id=f"tom_{uuid.uuid4().hex[:12]}",
                node_type="ToMHypothesisNode",
                topic_name=topic_name,
                hypothesis_type=item.get("hypothesis_type", "unknown"),
                label=item.get("label", "unknown"),
                hypothesis_text=item.get("hypothesis_text", ""),
                evidence_span_ids=item.get("evidence_span_ids", []),
                confidence=float(item.get("confidence", 0.0)),
                alternative_explanations=item.get("alternative_explanations", []),
                missing_information=item.get("missing_information", []),
                verification_status=item.get("verification_status", "unverified"),
            )
            topic.hypotheses.append(node)
            self.edges.append({"source": topic_name, "target": node.node_id, "relation": "has_hypothesis"})
            for evidence_id in node.evidence_span_ids:
                self.edges.append({"source": node.node_id, "target": evidence_id, "relation": "supported_by"})
            created.append(node)
        return created

    def update_sufficiency(self, topic_name: str, sufficiency: SufficiencyVector) -> None:
        self.ensure_topic(topic_name).sufficiency = sufficiency

    def set_score(self, topic_name: str, score: TopicScore) -> None:
        topic = self.ensure_topic(topic_name)
        topic.score = score
        topic.status = "abstained" if score.abstained else "completed"
        self.edges.append({"source": topic_name, "target": score.node_id, "relation": "summarized_into"})

    def set_pattern_tags(self, topic_name: str, tags: list[str]) -> None:
        self.ensure_topic(topic_name).pattern_tags = sorted(set(tags))

    def topic_snapshot(self, topic_name: str) -> dict[str, Any]:
        topic = self.ensure_topic(topic_name)
        return {
            "topic_name": topic.topic_name,
            "status": topic.status,
            "turn_count": len(topic.turns),
            "active_hypothesis_ids": [item.node_id for item in topic.hypotheses if item.verification_status == "unverified"],
            "sufficiency": topic.sufficiency.to_dict(),
            "statements": [item.to_dict() for item in topic.statements],
            "hypotheses": [item.to_dict() for item in topic.hypotheses],
            "score": topic.score.to_dict() if topic.score else None,
            "pattern_tags": topic.pattern_tags,
        }

    def export(self) -> dict[str, Any]:
        return {
            "participant_info": self.participant_info,
            "topics": {name: topic.to_dict() for name, topic in self.topics.items()},
            "edges": list(self.edges),
        }


def derive_pattern_tags(topic_state: TopicState) -> list[str]:
    tags: set[str] = set()
    statements = topic_state.statements
    if any(statement.extracted_evidence.symptom for statement in statements):
        tags.add("explicit_symptom_present")
    if not any(statement.extracted_evidence.frequency for statement in statements):
        tags.add("frequency_missing")
    else:
        tags.add("frequency_present")
    if any(statement.extracted_evidence.impact for statement in statements):
        tags.add("impact_present")
    else:
        tags.add("impact_missing")
    if any(statement.extracted_evidence.time_frame for statement in statements):
        tags.add("time_present")
    else:
        tags.add("time_missing")
    if any(statement.extracted_evidence.social_belief for statement in statements):
        tags.add("social_belief_signal")
    if any(statement.extracted_evidence.self_belief for statement in statements):
        tags.add("negative_self_belief_signal")
    if any("possible_minimization" in statement.extracted_evidence.inconsistency_flags for statement in statements):
        tags.add("possible_minimization")
    if any(statement.extracted_evidence.frequency and statement.extracted_evidence.impact for statement in statements):
        tags.add("frequency_and_impact_present")
    if any(
        "several_days" in statement.extracted_evidence.frequency or "rarely" in statement.extracted_evidence.frequency
        for statement in statements
    ) and any(statement.extracted_evidence.impact for statement in statements):
        tags.add("impact_high_frequency_low_candidate")
    if any(hyp.label == "possible_symptom_minimization" for hyp in topic_state.hypotheses):
        tags.add("possible_minimization")
    if any(hyp.verification_status == "unverified" for hyp in topic_state.hypotheses):
        tags.add("tom_unverified")
    return sorted(tags)

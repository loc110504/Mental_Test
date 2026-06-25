from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DictMixin:
    def to_dict(self) -> dict[str, Any]:
        return _convert(asdict(self))


def _convert(value: Any) -> Any:
    if is_dataclass(value):
        return _convert(asdict(value))
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


@dataclass
class ParticipantProfile(DictMixin):
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    source: str = "unknown"


@dataclass
class TranscriptTurn(DictMixin):
    turn_id: str
    role_name: str
    content: str


@dataclass
class StatementEvidence(DictMixin):
    symptom: list[str] = field(default_factory=list)
    frequency: list[str] = field(default_factory=list)
    time_frame: list[str] = field(default_factory=list)
    impact: list[str] = field(default_factory=list)
    emotion: list[str] = field(default_factory=list)
    self_belief: list[str] = field(default_factory=list)
    social_belief: list[str] = field(default_factory=list)
    inconsistency_flags: list[str] = field(default_factory=list)


@dataclass
class Provenance(DictMixin):
    source_type: str
    source_span_ids: list[str] = field(default_factory=list)


@dataclass
class StatementNode(DictMixin):
    node_id: str
    node_type: str
    topic_name: str
    speaker: str
    turn_id: int
    raw_text: str
    extracted_evidence: StatementEvidence
    provenance: Provenance


@dataclass
class ToMHypothesis(DictMixin):
    node_id: str
    node_type: str
    topic_name: str
    hypothesis_type: str
    label: str
    hypothesis_text: str
    evidence_span_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    alternative_explanations: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    verification_status: str = "unverified"


@dataclass
class SufficiencyVector(DictMixin):
    presence: float = 0.0
    time: float = 0.0
    frequency: float = 0.0
    impact: float = 0.0
    consistency: float = 1.0
    tom_verification: float = 1.0


@dataclass
class TopicScore(DictMixin):
    node_id: str
    node_type: str
    topic_name: str
    predicted_score: int | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    uncertainty: float = 0.0
    frequency_evidence: list[str] = field(default_factory=list)
    impact_evidence: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    used_lesson_ids: list[str] = field(default_factory=list)
    rubric_checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionTurn(DictMixin):
    turn_index: int
    question_text: str
    question_type: str
    target_gaps: list[str] = field(default_factory=list)
    verifies_hypothesis_ids: list[str] = field(default_factory=list)
    utility_breakdown: dict[str, float] = field(default_factory=dict)
    answer_text: str = ""
    answer_provenance: Provenance | None = None
    fidelity_score: float | None = None


@dataclass
class TopicState(DictMixin):
    topic_name: str
    status: str = "in_progress"
    turns: list[QuestionTurn] = field(default_factory=list)
    statements: list[StatementNode] = field(default_factory=list)
    hypotheses: list[ToMHypothesis] = field(default_factory=list)
    sufficiency: SufficiencyVector = field(default_factory=SufficiencyVector)
    score: TopicScore | None = None
    retrieved_lessons: dict[str, list[str]] = field(default_factory=dict)
    pattern_tags: list[str] = field(default_factory=list)


@dataclass
class GroundTruth(DictMixin):
    item_scores: dict[str, int] = field(default_factory=dict)
    total_score: int | None = None
    binary_label: int | None = None


@dataclass
class CaseRecord(DictMixin):
    schema_version: str
    case_id: str
    mode: str
    scale_name: str
    participant_profile: ParticipantProfile
    transcript: list[TranscriptTurn] = field(default_factory=list)
    topic_order: list[str] = field(default_factory=list)
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult(DictMixin):
    ground_truth_score: int
    predicted_score: int | None
    absolute_error: int
    normalized_reward: float


@dataclass
class AgentAssessment(DictMixin):
    agent_role: str
    quality: str
    responsibility: float
    diagnosis: str


@dataclass
class AccountabilityReport(DictMixin):
    case_id: str
    topic_name: str
    predicted_score: int | None
    ground_truth_score: int
    absolute_error: int
    outcome_type: str
    agent_assessments: list[AgentAssessment] = field(default_factory=list)


@dataclass
class Lesson(DictMixin):
    schema_version: str
    lesson_id: str
    agent_role: str
    phq_item: str
    pattern_tags: list[str]
    outcome_type: str
    diagnosis: str
    recommended_action: str
    action_to_avoid: str
    evidence: dict[str, Any]
    confidence: float
    usage_count: int = 0
    support_count: int = 1
    success_count: int = 0
    failure_count: int = 0
    historical_success_rate: float = 0.5
    status: str = "active"
    created_at: str = field(default_factory=utc_now_iso)
    last_updated: str = field(default_factory=utc_now_iso)
    last_used_at: str | None = None
    quarantine_reason: str | None = None


@dataclass
class CaseRunResult(DictMixin):
    case_id: str
    total_score: int
    binary_label: int
    topic_scores: dict[str, int | None]
    abstained_topics: list[str] = field(default_factory=list)
    report_summary: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class RunConfig(DictMixin):
    repo_root: str
    mode: str = "offline_train"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    temperature: float = 0.0
    max_completion_tokens: int = 1200
    max_turns_per_topic: int = 4
    question_candidates: int = 4
    experience_top_k: int = 5
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 1.5
    artifacts_root: str = "artifacts/evo"
    experience_root: str = "artifacts/evo/experiences"
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    interactive: bool = False
    allow_experience_updates: bool = True

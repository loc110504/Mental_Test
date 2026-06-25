from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..core.schemas import Lesson, utc_now_iso
from ..core.serialization import append_jsonl, read_jsonl, write_json


def _lesson_key(agent_role: str, phq_item: str, diagnosis: str, recommended_action: str) -> str:
    raw = "||".join([agent_role, phq_item, diagnosis.strip(), recommended_action.strip()])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class ExperienceTree:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_path = self.root / "active_lessons.jsonl"
        self.quarantine_path = self.root / "quarantine_lessons.jsonl"
        self.consolidation_log_path = self.root / "consolidation_log.jsonl"
        self.snapshot_path = self.root / "experience_snapshot.json"
        self.active_lessons = self._load(self.active_path)
        self.quarantine_lessons = self._load(self.quarantine_path)
        self._persist_snapshot()

    def _load(self, path: Path) -> dict[str, Lesson]:
        lessons: dict[str, Lesson] = {}
        for row in read_jsonl(path):
            lesson = Lesson(**row)
            lessons[lesson.lesson_id] = lesson
        return lessons

    def retrieve(
        self,
        agent_role: str,
        topic_name: str,
        pattern_tags: list[str],
        max_k: int,
        exclude_ids: list[str] | None = None,
    ) -> list[Lesson]:
        traces = self.retrieve_with_scores(agent_role, topic_name, pattern_tags, max_k, exclude_ids)
        return [item["lesson"] for item in traces]

    def retrieve_with_scores(
        self,
        agent_role: str,
        topic_name: str,
        pattern_tags: list[str],
        max_k: int,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        exclude = set(exclude_ids or [])
        target_tags = set(pattern_tags)
        scored: list[dict[str, Any]] = []
        for lesson in self.active_lessons.values():
            if lesson.lesson_id in exclude or lesson.status != "active":
                continue
            role_match = 1.0 if lesson.agent_role == agent_role else 0.0
            if role_match == 0.0:
                continue
            topic_match = 1.0 if lesson.phq_item == topic_name else 0.0
            if topic_match == 0.0:
                continue
            lesson_tags = set(lesson.pattern_tags)
            overlap = len(target_tags & lesson_tags)
            union = len(target_tags | lesson_tags) or 1
            pattern_similarity = overlap / union
            confidence = lesson.confidence
            historical_utility = lesson.historical_success_rate
            contradiction_risk = 0.0 if lesson.status == "active" else 1.0
            score = (
                0.25 * role_match
                + 0.20 * topic_match
                + 0.20 * pattern_similarity
                + 0.15 * confidence
                + 0.10 * historical_utility
                + 0.05 * min(lesson.usage_count / 10.0, 1.0)
                - 0.05 * contradiction_risk
            )
            scored.append(
                {
                    "lesson": lesson,
                    "score": score,
                    "breakdown": {
                        "role_match": role_match,
                        "topic_match": topic_match,
                        "pattern_similarity": pattern_similarity,
                        "confidence": confidence,
                        "historical_utility": historical_utility,
                        "usage_bonus": min(lesson.usage_count / 10.0, 1.0),
                        "contradiction_risk": contradiction_risk,
                    },
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:max_k]

    def store_lessons(self, validated_lessons: list[dict[str, Any]]) -> list[Lesson]:
        stored: list[Lesson] = []
        for row in validated_lessons:
            lesson_data = dict(row["lesson"])
            lesson_id = _lesson_key(
                lesson_data["agent_role"],
                lesson_data["phq_item"],
                lesson_data["diagnosis"],
                lesson_data["recommended_action"],
            )
            existing = self.active_lessons.get(lesson_id) or self.quarantine_lessons.get(lesson_id)
            status = row["decision"]
            target_status = "active" if status == "store_active" else "quarantine"
            if existing:
                existing.confidence = max(existing.confidence, float(lesson_data.get("confidence", existing.confidence)))
                existing.support_count += 1
                existing.last_updated = utc_now_iso()
                existing.status = target_status
                existing.quarantine_reason = row.get("rationale") if target_status == "quarantine" else None
                existing.pattern_tags = sorted(set(existing.pattern_tags + lesson_data.get("pattern_tags", [])))
                existing.evidence.setdefault("supporting_cases", [])
                case_id = lesson_data.get("evidence", {}).get("case_id")
                if case_id and case_id not in existing.evidence["supporting_cases"]:
                    existing.evidence["supporting_cases"].append(case_id)
                lesson = existing
            else:
                lesson = Lesson(
                    schema_version="1.0",
                    lesson_id=lesson_id,
                    agent_role=lesson_data["agent_role"],
                    phq_item=lesson_data["phq_item"],
                    pattern_tags=lesson_data["pattern_tags"],
                    outcome_type=lesson_data["outcome_type"],
                    diagnosis=lesson_data["diagnosis"],
                    recommended_action=lesson_data["recommended_action"],
                    action_to_avoid=lesson_data["action_to_avoid"],
                    evidence=lesson_data["evidence"],
                    confidence=float(lesson_data.get("confidence", 0.5)),
                    status=target_status,
                    quarantine_reason=row.get("rationale") if target_status == "quarantine" else None,
                )
            if lesson.status == "active":
                self.active_lessons[lesson.lesson_id] = lesson
                self.quarantine_lessons.pop(lesson.lesson_id, None)
            else:
                self.quarantine_lessons[lesson.lesson_id] = lesson
                self.active_lessons.pop(lesson.lesson_id, None)
            stored.append(lesson)
        self._persist_snapshot()
        return stored

    def register_usage(self, lesson_ids: list[str]) -> None:
        timestamp = utc_now_iso()
        changed = False
        for lesson_id in lesson_ids:
            lesson = self.active_lessons.get(lesson_id)
            if not lesson:
                continue
            lesson.usage_count += 1
            lesson.last_used_at = timestamp
            lesson.last_updated = timestamp
            changed = True
        if changed:
            self._persist_snapshot()

    def apply_outcome_feedback(self, lesson_ids: list[str], normalized_reward: float) -> None:
        changed = False
        for lesson_id in lesson_ids:
            lesson = self.active_lessons.get(lesson_id)
            if not lesson:
                continue
            if normalized_reward >= 0.67:
                lesson.success_count += 1
            else:
                lesson.failure_count += 1
            total = lesson.success_count + lesson.failure_count
            if total > 0:
                lesson.historical_success_rate = lesson.success_count / total
            lesson.last_updated = utc_now_iso()
            changed = True
        if changed:
            self._persist_snapshot()

    def release_quarantine_lessons(self, min_support_count: int = 3, min_confidence: float = 0.75) -> list[Lesson]:
        released: list[Lesson] = []
        for lesson_id, lesson in list(self.quarantine_lessons.items()):
            if lesson.support_count >= min_support_count and lesson.confidence >= min_confidence:
                lesson.status = "active"
                lesson.quarantine_reason = None
                lesson.last_updated = utc_now_iso()
                self.active_lessons[lesson_id] = lesson
                del self.quarantine_lessons[lesson_id]
                released.append(lesson)
        if released:
            append_jsonl(
                self.consolidation_log_path,
                {
                    "event": "release_quarantine_lessons",
                    "released_ids": [lesson.lesson_id for lesson in released],
                    "timestamp": utc_now_iso(),
                },
            )
            self._persist_snapshot()
        return released

    def consolidate_duplicates(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str, tuple[str, ...]], list[Lesson]] = {}
        for lesson in self.active_lessons.values():
            key = (lesson.agent_role, lesson.phq_item, tuple(sorted(lesson.pattern_tags)))
            groups.setdefault(key, []).append(lesson)

        merged_events: list[dict[str, Any]] = []
        for _, lessons in groups.items():
            if len(lessons) < 2:
                continue
            canonical = lessons[0]
            for lesson in lessons[1:]:
                if (
                    lesson.diagnosis == canonical.diagnosis
                    and lesson.recommended_action == canonical.recommended_action
                ):
                    canonical.support_count += lesson.support_count
                    canonical.usage_count += lesson.usage_count
                    canonical.success_count += lesson.success_count
                    canonical.failure_count += lesson.failure_count
                    total = canonical.success_count + canonical.failure_count
                    canonical.historical_success_rate = canonical.success_count / total if total else canonical.historical_success_rate
                    canonical.confidence = max(canonical.confidence, lesson.confidence)
                    canonical.last_updated = utc_now_iso()
                    del self.active_lessons[lesson.lesson_id]
                    merged_events.append(
                        {
                            "event": "merge_duplicate_lessons",
                            "canonical_id": canonical.lesson_id,
                            "merged_id": lesson.lesson_id,
                            "timestamp": utc_now_iso(),
                        }
                    )
        for event in merged_events:
            append_jsonl(self.consolidation_log_path, event)
        if merged_events:
            self._persist_snapshot()
        return merged_events

    def _persist_snapshot(self) -> None:
        write_json(
            self.snapshot_path,
            {
                "active_lessons": [lesson.to_dict() for lesson in self.active_lessons.values()],
                "quarantine_lessons": [lesson.to_dict() for lesson in self.quarantine_lessons.values()],
            },
        )
        self._rewrite_jsonl(self.active_path, self.active_lessons)
        self._rewrite_jsonl(self.quarantine_path, self.quarantine_lessons)

    @staticmethod
    def _rewrite_jsonl(path: Path, lessons: dict[str, Lesson]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for lesson in lessons.values():
                handle.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")

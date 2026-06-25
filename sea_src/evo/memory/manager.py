from __future__ import annotations

from typing import Any

from ..core.schemas import utc_now_iso
from ..core.serialization import append_jsonl


class MemoryManager:
    def __init__(self, experience_tree, logger) -> None:
        self.experience_tree = experience_tree
        self.logger = logger

    def record_retrieval_feedback(self, lesson_ids: list[str], normalized_reward: float) -> None:
        if not lesson_ids:
            return
        self.experience_tree.apply_outcome_feedback(lesson_ids, normalized_reward)

    def register_retrieval(self, lesson_ids: list[str]) -> None:
        if not lesson_ids:
            return
        self.experience_tree.register_usage(lesson_ids)

    def maintain(self) -> dict[str, Any]:
        merged = self.experience_tree.consolidate_duplicates()
        released = self.experience_tree.release_quarantine_lessons()
        event = {
            "event": "memory_maintenance",
            "merged_count": len(merged),
            "released_count": len(released),
            "timestamp": utc_now_iso(),
        }
        append_jsonl(self.experience_tree.consolidation_log_path, event)
        return {
            "merged": merged,
            "released": [lesson.to_dict() for lesson in released],
            "event": event,
        }

"""Per-model mutable state kept outside checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DQECGRuntimeState:
    schedule: Any = None
    visual_start: Optional[int] = None
    visual_length: Optional[int] = None
    original_visual_length: Optional[int] = None
    visual_positions: Any = None
    visual_token_counts: Any = None
    pivot_indices: Any = None
    patch_assignments: Any = None
    current_visual_indices: Any = None
    question_positions: Any = None
    original_question_positions: Any = None
    cache_active: bool = False
    next_position_id: Optional[int] = None
    layer_first_indices: Any = None
    layer_second_indices: Any = None
    final_visual_indices: Any = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def clear_question_positions(self) -> None:
        """End the current sample's question-position lifecycle."""

        self.question_positions = None
        self.original_question_positions = None

    def reset_prefill(self) -> None:
        question = self.original_question_positions
        if question is None:
            question = self.question_positions
            if hasattr(question, "detach"):
                question = question.detach().clone()
            self.original_question_positions = question
        elif hasattr(question, "clone"):
            question = question.clone()
        self.schedule = None
        self.visual_start = None
        self.visual_length = None
        self.original_visual_length = None
        self.visual_positions = None
        self.visual_token_counts = None
        self.pivot_indices = None
        self.patch_assignments = None
        self.current_visual_indices = None
        self.question_positions = question
        self.cache_active = False
        self.next_position_id = None
        self.layer_first_indices = None
        self.layer_second_indices = None
        self.final_visual_indices = None
        self.debug = {}

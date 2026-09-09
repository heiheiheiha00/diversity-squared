"""Small shared policy interface consumed by the decoder runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..config import DQECGSchedule


@dataclass(frozen=True)
class DecoderPolicy:
    """Architecture-specific entropy windows around fixed prune boundaries."""

    architecture: str
    first_entropy_offsets: Tuple[int, ...]
    second_entropy_offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.first_entropy_offsets or not self.second_entropy_offsets:
            raise ValueError("Both entropy windows must contain at least one layer.")
        if tuple(sorted(set(self.first_entropy_offsets))) != self.first_entropy_offsets:
            raise ValueError("First-stage entropy offsets must be sorted and unique.")
        if tuple(sorted(set(self.second_entropy_offsets))) != self.second_entropy_offsets:
            raise ValueError("Second-stage entropy offsets must be sorted and unique.")
        if self.first_entropy_offsets[-1] != 0:
            raise ValueError("First-stage entropy window must end at its prune layer.")
        if self.second_entropy_offsets[-1] != 0:
            raise ValueError("Second-stage entropy window must end at its prune layer.")

    @staticmethod
    def _resolve(boundary: int, offsets: Tuple[int, ...], layer_count: int):
        layers = tuple(int(boundary) + offset for offset in offsets)
        if layers[0] < 0 or layers[-1] >= int(layer_count):
            raise ValueError(
                f"Entropy window {layers} falls outside a {layer_count}-layer decoder."
            )
        return layers

    def first_entropy_layers(self, schedule: "DQECGSchedule") -> Tuple[int, ...]:
        return self._resolve(
            schedule.first_boundary,
            self.first_entropy_offsets,
            schedule.layer_count,
        )

    def second_entropy_layers(self, schedule: "DQECGSchedule") -> Tuple[int, ...]:
        return self._resolve(
            schedule.second_boundary,
            self.second_entropy_offsets,
            schedule.layer_count,
        )

    def capture_layers(self, schedule: "DQECGSchedule") -> Tuple[int, ...]:
        return tuple(
            sorted(
                set(self.first_entropy_layers(schedule))
                | set(self.second_entropy_layers(schedule))
            )
        )

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpecialistDescriptor:
    model_name: str
    capabilities: tuple[str, ...]
    required_modalities: tuple[str, ...]
    available: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "capabilities": list(self.capabilities),
            "required_modalities": list(self.required_modalities),
            "available": self.available,
            "reason": self.reason,
        }


class SpecialistModel(Protocol):
    model_name: str
    capabilities: tuple[str, ...]
    required_modalities: tuple[str, ...]

    def describe(self) -> SpecialistDescriptor:
        ...

from typing import Any, Iterable

from pydantic import BaseModel, Field

from backend.confidence.report import build_confidence_report
from backend.models.schemas import ConfidenceReport, Evidence


class EvidenceAggregation(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    confidence_report: ConfidenceReport


def aggregate_evidence(
    evidence: Iterable[Evidence],
    image_analysis: Iterable[dict[str, Any]] | None = None,
) -> EvidenceAggregation:
    preserved = list(evidence)
    findings = _unique_in_order(item.finding for item in preserved if item.finding)
    sources = _unique_in_order(
        item.model or item.source for item in preserved if item.model or item.source
    )
    provenance = [
        {
            "source": item.source,
            "model": item.model,
            "provenance": item.provenance,
        }
        for item in preserved
    ]
    return EvidenceAggregation(
        evidence=preserved,
        findings=findings,
        sources=sources,
        provenance=provenance,
        confidence_report=build_confidence_report(preserved, image_analysis),
    )


def _unique_in_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result

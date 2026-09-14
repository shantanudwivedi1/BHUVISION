from datetime import datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field


TaskName = Literal["vqa", "captioning", "change_detection", "object_detection", "unsupported"]
ModalityName = Literal["optical", "sar", "multispectral", "hyperspectral", "thermal", "radar"]
OutputType = Literal["answer", "caption", "object_list", "change_map", "analysis", "evidence", "detection_list"]
ConfidenceSignalStatus = Literal["available", "unavailable"]


class QueryPlan(BaseModel):
    task: TaskName
    objects: list[str] = Field(default_factory=list)
    image_count: int = Field(default=0, ge=0)
    temporal: bool = False
    modalities: list[ModalityName] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    output_types: list[OutputType] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    query: str = Field(min_length=1)


class Evidence(BaseModel):
    evidence_type: str
    source: str
    model: str | None = None
    finding: str | None = None
    spatial: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class ConfidenceSignal(BaseModel):
    name: str
    status: ConfidenceSignalStatus
    value: float | None = Field(default=None, ge=0, le=1)
    metrics: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class ConfidenceReport(BaseModel):
    signals: list[ConfidenceSignal] = Field(default_factory=list)
    aggregate_value: float | None = Field(default=None, ge=0, le=1)
    aggregation_method: str
    notes: list[str] = Field(default_factory=list)


class AuditEntry(BaseModel):
    analysis_id: str
    query: str
    plan: QueryPlan
    routing_decision: dict[str, Any]
    execution_status: str
    timestamp: datetime
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    query: str
    plan: QueryPlan
    routing: dict[str, Any]
    result: dict[str, Any]
    evidence: list[Evidence] = Field(default_factory=list)
    confidence_report: ConfidenceReport
    audit_trace: AuditEntry
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

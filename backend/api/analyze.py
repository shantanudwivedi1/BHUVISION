import json
import uuid
from email import policy
from email.parser import BytesParser

from fastapi import APIRouter, HTTPException, Request

from backend.audit.trace import create_audit_entry
from backend.evidence.aggregation import aggregate_evidence
from backend.evidence.image_analysis import UnsupportedImageError, inspect_image
from backend.models.schemas import AnalysisRequest, AnalysisResponse, Evidence
from backend.planner.planner import build_query_plan_with_optional_llm
from backend.router.model_router import route_plan

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: Request) -> AnalysisResponse:
    query, uploads = await _read_request(request)
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    request_model = AnalysisRequest(query=query)
    analysis_id = str(uuid.uuid4())
    plan = build_query_plan_with_optional_llm(request_model.query, len(uploads))
    routing = route_plan(plan)
    errors: list[str] = []
    warnings: list[str] = []
    image_analysis = []
    evidence = []

    for filename, content_type, data in uploads:
        try:
            info = inspect_image(data, filename)
            info["content_type"] = content_type
            image_analysis.append(info)
            evidence.append(
                Evidence(
                    evidence_type="image_metadata",
                    source=filename,
                    finding="Image format, dimensions, channels, and basic quality metrics measured.",
                    spatial={"image_dimensions": {"width": info["width"], "height": info["height"]}},
                    provenance={
                        "method": "lightweight_image_header_inspection",
                        "content_type": content_type,
                    },
                    details=info,
                )
            )
        except UnsupportedImageError as exc:
            errors.append(str(exc))

    if routing["requested_capability"] and not routing["available"]:
        warnings.append(
            f"{routing['requested_capability']} is unavailable; no semantic model execution was performed."
        )
    elif routing["requested_capability"]:
        warnings.append(
            f"{routing['requested_capability']} was not executed; this endpoint currently exposes lightweight metadata only."
        )
    if plan.task == "unsupported":
        warnings.append("Query pattern is unsupported by the deterministic planner.")
    if not uploads:
        warnings.append("No images were uploaded; image metadata analysis was skipped.")

    execution_status = "completed_lightweight_only"
    if errors and not image_analysis:
        execution_status = "completed_with_errors"
    audit = create_audit_entry(
        analysis_id, request_model.query, plan, routing, execution_status, errors, warnings
    )
    aggregation = aggregate_evidence(evidence, image_analysis)
    return AnalysisResponse(
        status="success" if not errors else "partial",
        analysis_id=analysis_id,
        query=request_model.query,
        plan=plan,
        routing=routing,
        result={
            "summary": "Lightweight image metadata analyzed; semantic satellite understanding was not performed.",
            "image_analysis": image_analysis,
            "evidence_summary": {
                "findings": aggregation.findings,
                "sources": aggregation.sources,
                "provenance": aggregation.provenance,
            },
        },
        evidence=aggregation.evidence,
        confidence_report=aggregation.confidence_report,
        audit_trace=audit,
        errors=errors,
        warnings=warnings,
    )


async def _read_request(request: Request) -> tuple[str, list[tuple[str, str, bytes]]]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    if content_type.startswith("application/json"):
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        return str(payload.get("query", "")).strip(), []
    if not content_type.startswith("multipart/form-data"):
        return request.query_params.get("query", "").strip(), []

    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    query = ""
    uploads: list[tuple[str, str, bytes]] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        content = part.get_payload(decode=True) or b""
        if name == "query" and not filename:
            query = content.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
        elif filename:
            uploads.append((filename, part.get_content_type(), content))
    return query, uploads

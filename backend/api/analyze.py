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
from backend.router.model_router import execute_plan, route_plan

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: Request) -> AnalysisResponse:
    query, uploads, modalities = await _read_request(request)
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    request_model = AnalysisRequest(query=query)
    analysis_id = str(uuid.uuid4())

    # Pass user-selected modalities into the planner.
    plan = build_query_plan_with_optional_llm(
        request_model.query, len(uploads), modalities=modalities or None,
    )
    routing = route_plan(plan)
    errors: list[str] = []
    warnings: list[str] = []
    image_analysis = []
    evidence = []

    # -- Lightweight image metadata inspection -----------------------------
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

    # -- Model execution (BLIP / YOLO / CROMA) ----------------------------
    model_evidence, model_warnings = execute_plan(plan, routing, uploads, request_model.query)
    evidence.extend(model_evidence)
    warnings.extend(model_warnings)

    # -- Warnings for unavailable capabilities ----------------------------
    if routing["requested_capability"] and not routing["available"]:
        warnings.append(
            f"{routing['requested_capability']} is unavailable; no semantic model execution was performed."
        )
    if plan.task == "unsupported":
        warnings.append("Query pattern is unsupported by the deterministic planner.")
    if not uploads:
        warnings.append("No images were uploaded; image metadata analysis was skipped.")

    # -- Determine execution status ----------------------------------------
    has_model_results = any(e.evidence_type == "model_finding" for e in evidence)
    if has_model_results:
        execution_status = "completed_with_model"
    elif errors and not image_analysis:
        execution_status = "completed_with_errors"
    else:
        execution_status = "completed_lightweight_only"

    audit = create_audit_entry(
        analysis_id, request_model.query, plan, routing, execution_status, errors, warnings
    )
    aggregation = aggregate_evidence(evidence, image_analysis)

    # Build the summary from model findings when available.
    model_findings = [e.finding for e in evidence if e.evidence_type == "model_finding" and e.finding]
    if model_findings:
        summary = " | ".join(model_findings)
    else:
        summary = "Lightweight image metadata analyzed; semantic satellite understanding was not performed."

    return AnalysisResponse(
        status="success" if not errors else "partial",
        analysis_id=analysis_id,
        query=request_model.query,
        plan=plan,
        routing=routing,
        result={
            "summary": summary,
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


# ---------------------------------------------------------------------------
# Request parsing — now also extracts the "modalities" form field
# ---------------------------------------------------------------------------

async def _read_request(request: Request) -> tuple[str, list[tuple[str, str, bytes]], list[str]]:
    """Return (query, uploads, modalities) from the incoming request."""
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if content_type.startswith("application/json"):
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        mods = _parse_modalities(payload.get("modalities", ""))
        return str(payload.get("query", "")).strip(), [], mods

    if not content_type.startswith("multipart/form-data"):
        return request.query_params.get("query", "").strip(), [], []

    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    query = ""
    modalities_raw = ""
    uploads: list[tuple[str, str, bytes]] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        content = part.get_payload(decode=True) or b""
        if name == "query" and not filename:
            query = content.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
        elif name == "modalities" and not filename:
            modalities_raw = content.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
        elif filename:
            uploads.append((filename, part.get_content_type(), content))
    return query, uploads, _parse_modalities(modalities_raw)


def _parse_modalities(raw: str | list | None) -> list[str]:
    """Parse a comma-separated modalities string or list into a clean list."""
    if isinstance(raw, list):
        return [m.strip() for m in raw if isinstance(m, str) and m.strip()]
    if isinstance(raw, str) and raw.strip():
        return [m.strip() for m in raw.split(",") if m.strip()]
    return []

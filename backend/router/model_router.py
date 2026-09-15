"""Route a QueryPlan to the right specialist model and execute it.

Specialists:
- GeoChat-7B (Primary remote sensing VLM for VQA, Captioning, and Grounded Detection)
- YOLO11 (Local real-time Object Detection with bounding boxes)
"""

from typing import Any

from backend.models.schemas import Evidence, QueryPlan
from backend.router.geochat import GeoChatAdapter
from backend.router.yolo import YOLOAdapter

# Singleton adapters — lightweight until first inference call.
_geochat = GeoChatAdapter()
_yolo = YOLOAdapter()


def route_plan(plan: QueryPlan) -> dict:
    """Describe which capability the plan needs and whether it's available."""

    geochat_desc = _geochat.describe().as_dict()

    # VQA / Captioning → GeoChat-7B only
    if plan.task in {"vqa", "captioning"}:
        return {
            "requested_capability": "GeoChat-7B",
            "model_name": "MBZUAI/geochat-7B",
            "capabilities": geochat_desc["capabilities"],
            "available": geochat_desc["available"],
            "execution": "not_executed",
            "reason": geochat_desc["reason"],
            "lightweight_capability": "image_metadata_analyzer",
            "lightweight_available": True,
        }

    # Object detection → GeoChat-7B grounding + YOLO11 bounding boxes
    if plan.task == "object_detection":
        model_names = []
        if geochat_desc["available"]:
            model_names.append("GeoChat-7B")
        yolo_desc = _yolo.describe().as_dict()
        if yolo_desc["available"]:
            model_names.append("YOLO11")

        return {
            "requested_capability": "GeoChat-7B + YOLO11",
            "model_name": " + ".join(model_names) or "YOLO11",
            "capabilities": ["object_detection", "grounding", "bounding_boxes"],
            "available": bool(model_names),
            "execution": "not_executed",
            "reason": "Grounded object detection with GeoChat-7B and real-time bounding boxes via YOLO11.",
            "lightweight_capability": "image_metadata_analyzer",
            "lightweight_available": True,
        }

    # Change detection (DeltaVLM) — stub for future integration
    if plan.task == "change_detection":
        return {
            "requested_capability": "DeltaVLM",
            "available": False,
            "execution": "not_executed",
            "reason": "DeltaVLM is not yet integrated; temporal change detection will be available in a future release.",
            "lightweight_capability": "image_metadata_analyzer",
            "lightweight_available": True,
        }

    # Unsupported / fallback
    return {
        "requested_capability": None,
        "available": False,
        "execution": "not_executed",
        "reason": "No supported capability matches this plan.",
        "lightweight_capability": "image_metadata_analyzer",
        "lightweight_available": True,
    }


# ---------------------------------------------------------------------------
# Helper: Separate Optical and SAR images from uploads
# ---------------------------------------------------------------------------

def _split_optical_and_sar(
    uploads: list[tuple[str, str, bytes]]
) -> tuple[list[tuple[str, str, bytes]], bytes | None]:
    """Identify SAR imagery from uploads (by filename or metadata)."""
    optical_uploads = []
    sar_bytes = None

    for filename, ct, data in uploads:
        lower_name = filename.lower()
        if any(tag in lower_name for tag in ("sar", "vh", "vv", "sentinel1", "s1")):
            sar_bytes = data
        else:
            optical_uploads.append((filename, ct, data))

    # If all uploads were tagged as SAR or none, treat the first as optical
    if not optical_uploads and uploads:
        optical_uploads = [uploads[0]]
        if len(uploads) > 1:
            sar_bytes = uploads[1][2]

    return optical_uploads, sar_bytes


# ---------------------------------------------------------------------------
# Model execution — called by analyze.py after routing
# ---------------------------------------------------------------------------

def execute_plan(
    plan: QueryPlan,
    routing: dict,
    uploads: list[tuple[str, str, bytes]],
    query: str,
) -> tuple[list[Evidence], list[str], list[dict[str, Any]]]:
    """Run the routed model on uploaded images.

    Returns (evidence_list, warnings_list, annotated_images).
    annotated_images is a list of {"filename": str, "data_url": str} dicts.
    """
    evidence: list[Evidence] = []
    warnings: list[str] = []
    annotated_images: list[dict[str, Any]] = []

    if not routing.get("available") or not uploads:
        return evidence, warnings, annotated_images

    optical_uploads, sar_bytes = _split_optical_and_sar(uploads)

    # -- VQA Task ----------------------------------------------------------
    if plan.task == "vqa":
        for filename, _ct, data in optical_uploads:
            try:
                result = _geochat.vqa(data, query, sar_bytes=sar_bytes)
                evidence.append(Evidence(
                    evidence_type="model_finding",
                    source=filename,
                    model=result["model"],
                    finding=result["answer"],
                    provenance={"method": "geochat_vqa", "question": query, "has_sar": sar_bytes is not None},
                ))
            except Exception as exc:
                warnings.append(f"GeoChat VQA failed on {filename}: {exc}")

    # -- Captioning Task ---------------------------------------------------
    elif plan.task == "captioning":
        for filename, _ct, data in optical_uploads:
            try:
                result = _geochat.caption(data, sar_bytes=sar_bytes)
                evidence.append(Evidence(
                    evidence_type="model_finding",
                    source=filename,
                    model=result["model"],
                    finding=result["caption"],
                    provenance={"method": "geochat_captioning", "has_sar": sar_bytes is not None},
                ))
            except Exception as exc:
                warnings.append(f"GeoChat captioning failed on {filename}: {exc}")

    # -- Object Detection Task ---------------------------------------------
    elif plan.task == "object_detection":
        for filename, _ct, data in optical_uploads:
            # 1. Run GeoChat Grounded Detection
            if _geochat.describe().available:
                try:
                    gc_result = _geochat.detect(data, query=query, sar_bytes=sar_bytes)
                    if gc_result.get("description"):
                        evidence.append(Evidence(
                            evidence_type="model_finding",
                            source=filename,
                            model=gc_result["model"],
                            finding=gc_result["description"],
                            provenance={"method": "geochat_grounding", "target": query},
                        ))
                    for det in gc_result.get("detections", []):
                        evidence.append(Evidence(
                            evidence_type="model_finding",
                            source=filename,
                            model=gc_result["model"],
                            finding=f"Located {det['class']}",
                            confidence=det["confidence"],
                            spatial={"bbox": det.get("bbox", det.get("box_pct"))},
                            provenance={"method": "geochat_bounding_box"},
                        ))
                    if gc_result.get("annotated_image"):
                        annotated_images.append({
                            "filename": f"geochat_{filename}",
                            "data_url": gc_result["annotated_image"],
                            "source_model": "GeoChat-7B",
                        })
                except Exception as exc:
                    warnings.append(f"GeoChat detection grounding failed on {filename}: {exc}")

            # 2. Run YOLO Object Detection (real-time bounding boxes)
            try:
                yolo_result = _yolo.detect(data)
                for det in yolo_result["detections"]:
                    evidence.append(Evidence(
                        evidence_type="model_finding",
                        source=filename,
                        model=yolo_result["model"],
                        finding=f"Detected {det['class']}",
                        confidence=det["confidence"],
                        spatial={"bbox": det["bbox"]},
                        provenance={"method": "yolo11_detection"},
                    ))
                if yolo_result.get("annotated_image"):
                    annotated_images.append({
                        "filename": f"yolo_{filename}",
                        "data_url": yolo_result["annotated_image"],
                        "source_model": "YOLO11",
                    })
                if not yolo_result["detections"] and not annotated_images:
                    warnings.append(f"No objects detected in {filename} above the confidence threshold.")
            except Exception as exc:
                warnings.append(f"YOLO detection failed on {filename}: {exc}")

    # Mark that the model was executed in the routing dict.
    if evidence:
        routing["execution"] = "executed"

    return evidence, warnings, annotated_images

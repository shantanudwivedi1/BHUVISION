"""Route a QueryPlan to the right specialist model and execute it.

Only two specialists: BLIP (VQA + Captioning) and YOLOv8 (Object Detection).
Adapters are instantiated once at module level (cheap — no model weights
are loaded until actual inference).
"""

from typing import Any

from backend.models.schemas import Evidence, QueryPlan
from backend.router.blip import BLIPAdapter
from backend.router.yolo import YOLOAdapter


# Singleton adapters — lightweight until first inference call.
_blip = BLIPAdapter()
_yolo = YOLOAdapter()


def route_plan(plan: QueryPlan) -> dict:
    """Describe which capability the plan needs and whether it's available."""

    # VQA / Captioning → BLIP
    if plan.task in {"vqa", "captioning"}:
        descriptor = _blip.describe().as_dict()
        return {
            "requested_capability": "BLIP",
            "model_name": descriptor["model_name"],
            "capabilities": descriptor["capabilities"],
            "available": descriptor["available"],
            "execution": "not_executed",
            "reason": descriptor["reason"],
            "lightweight_capability": "image_metadata_analyzer",
            "lightweight_available": True,
        }

    # Object detection → YOLO
    if plan.task == "object_detection":
        descriptor = _yolo.describe().as_dict()
        return {
            "requested_capability": "YOLOv8",
            "model_name": descriptor["model_name"],
            "capabilities": descriptor["capabilities"],
            "available": descriptor["available"],
            "execution": "not_executed",
            "reason": descriptor["reason"],
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
    annotated_images is a list of {"filename": str, "data_url": str} dicts
    containing base64-encoded PNGs with bounding boxes drawn on them.
    """
    evidence: list[Evidence] = []
    warnings: list[str] = []
    annotated_images: list[dict[str, Any]] = []

    if not routing.get("available") or not uploads:
        return evidence, warnings, annotated_images

    # -- BLIP VQA ----------------------------------------------------------
    if plan.task == "vqa":
        for filename, _ct, data in uploads:
            try:
                result = _blip.vqa(data, query)
                evidence.append(Evidence(
                    evidence_type="model_finding",
                    source=filename,
                    model=result["model"],
                    finding=result["answer"],
                    provenance={"method": "blip_vqa", "question": query},
                ))
            except Exception as exc:
                warnings.append(f"BLIP VQA failed on {filename}: {exc}")

    # -- BLIP Captioning ---------------------------------------------------
    elif plan.task == "captioning":
        for filename, _ct, data in uploads:
            try:
                result = _blip.caption(data)
                evidence.append(Evidence(
                    evidence_type="model_finding",
                    source=filename,
                    model=result["model"],
                    finding=result["caption"],
                    provenance={"method": "blip_captioning"},
                ))
            except Exception as exc:
                warnings.append(f"BLIP captioning failed on {filename}: {exc}")

    # -- YOLO Object Detection ---------------------------------------------
    elif plan.task == "object_detection":
        for filename, _ct, data in uploads:
            try:
                result = _yolo.detect(data)
                for det in result["detections"]:
                    evidence.append(Evidence(
                        evidence_type="model_finding",
                        source=filename,
                        model=result["model"],
                        finding=f"Detected {det['class']}",
                        confidence=det["confidence"],
                        spatial={"bbox": det["bbox"]},
                        provenance={"method": "yolov8_detection"},
                    ))
                # Attach the annotated image with bounding boxes drawn
                if result.get("annotated_image"):
                    annotated_images.append({
                        "filename": filename,
                        "data_url": result["annotated_image"],
                    })
                if not result["detections"]:
                    warnings.append(f"No objects detected in {filename} above the confidence threshold.")
            except Exception as exc:
                warnings.append(f"YOLO detection failed on {filename}: {exc}")

    # Mark that the model was executed in the routing dict.
    if evidence:
        routing["execution"] = "executed"

    return evidence, warnings, annotated_images

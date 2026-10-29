"""Route analysis tasks to locally installed specialist models only."""

from typing import Any

from backend.models.schemas import Evidence, QueryPlan
from backend.router.blip import BLIPAdapter
from backend.router.grounding_dino import GroundingDINOAdapter

# Singleton adapters — lightweight until first inference call.
_blip = BLIPAdapter()
_grounding_dino = GroundingDINOAdapter()


def route_plan(plan: QueryPlan) -> dict:
    """Describe which capability the plan needs and whether it's available."""

    blip_desc = _blip.describe().as_dict()

    # VQA and captioning use locally cached BLIP weights.
    if plan.task in {"vqa", "captioning"}:
        return {
            "requested_capability": "Local BLIP",
            "model_name": "Salesforce/blip-vqa-base",
            "capabilities": blip_desc["capabilities"],
            "available": blip_desc["available"],
            "execution": "not_executed",
            "reason": blip_desc["reason"],
            "lightweight_capability": "image_metadata_analyzer",
            "lightweight_available": True,
        }

    # Object detection uses locally cached Grounding DINO.
    if plan.task == "object_detection":
        dino_desc = _grounding_dino.describe().as_dict()
        return {
            "requested_capability": "Local Grounding DINO",
            "model_name": "Grounding DINO Tiny" if dino_desc["available"] else "",
            "capabilities": ["object_detection", "grounding", "open_vocabulary", "bounding_boxes"],
            "available": dino_desc["available"],
            "execution": "not_executed",
            "reason": dino_desc["reason"],
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
                result = _blip.vqa(data, query)
                evidence.append(Evidence(
                    evidence_type="model_finding",
                    source=filename,
                    model=result["model"],
                    finding=result["answer"],
                    provenance={"method": "local_blip_vqa", "question": query},
                ))
            except Exception as exc:
                warnings.append(f"Local BLIP VQA failed on {filename}: {exc}")

    # -- Captioning Task ---------------------------------------------------
    elif plan.task == "captioning":
        for filename, _ct, data in optical_uploads:
            try:
                result = _blip.caption(data)
                evidence.append(Evidence(
                    evidence_type="model_finding",
                    source=filename,
                    model=result["model"],
                    finding=result["caption"],
                    provenance={"method": "local_blip_captioning"},
                ))
            except Exception as exc:
                warnings.append(f"Local BLIP captioning failed on {filename}: {exc}")

    # -- Object Detection Task ---------------------------------------------
    elif plan.task == "object_detection":
        for filename, _ct, data in optical_uploads:
            try:
                dino_result = _grounding_dino.detect(data, query=query)
                for det in dino_result["detections"]:
                    evidence.append(Evidence(
                        evidence_type="model_finding",
                        source=filename,
                        model=dino_result["model"],
                        finding=f"Detected {det['class']}",
                        confidence=det["confidence"],
                        spatial={"bbox": det["bbox"]},
                        provenance={"method": "grounding_dino_detection", "prompt": dino_result.get("text_prompt")},
                    ))
                if dino_result.get("annotated_image"):
                    annotated_images.append({
                        "filename": f"detection_{filename}",
                        "data_url": dino_result["annotated_image"],
                        "source_model": "Grounding DINO",
                    })
                if not dino_result["detections"] and not annotated_images:
                    warnings.append(f"No objects detected in {filename} matching query '{query}' above confidence threshold.")
            except Exception as exc:
                warnings.append(f"Local object detection failed on {filename}: {exc}")

    # Mark that the model was executed in the routing dict.
    if evidence:
        routing["execution"] = "executed"

    return evidence, warnings, annotated_images

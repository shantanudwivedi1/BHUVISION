from backend.models.schemas import QueryPlan
from backend.router.croma import CROMAAdapter


CAPABILITIES = {
    "vqa": {"capability": "RS/VLM capability", "available": False},
    "captioning": {"capability": "RS/VLM capability", "available": False},
    "change_detection": {"capability": "DeltaVLM", "available": False},
    "optical_sar_analysis": {"capability": "CROMA", "available": False},
    "unsupported": {"capability": None, "available": False},
}


def route_plan(plan: QueryPlan) -> dict:
    if plan.task == "optical_sar_analysis":
        descriptor = CROMAAdapter().describe().as_dict()
        return {
            "requested_capability": "CROMA",
            "model_name": descriptor["model_name"],
            "capabilities": descriptor["capabilities"],
            "required_modalities": descriptor["required_modalities"],
            "available": descriptor["available"],
            "execution": "not_executed",
            "output_type": "learned_representation/features",
            "reason": (
                descriptor["reason"]
                if not descriptor["available"]
                else "CROMA is available for feature extraction; modality-labeled Sentinel-1/2 tensors are required for execution."
            ),
            "lightweight_capability": "image_metadata_analyzer",
            "lightweight_available": True,
        }
    selected = CAPABILITIES.get(plan.task, CAPABILITIES["unsupported"])
    return {
        "requested_capability": selected["capability"],
        "available": selected["available"],
        "execution": "not_executed",
        "reason": (
            "Semantic model integration is not installed; only lightweight image metadata was analyzed."
            if selected["capability"]
            else "No supported capability matches this plan."
        ),
        "lightweight_capability": "image_metadata_analyzer",
        "lightweight_available": True,
    }

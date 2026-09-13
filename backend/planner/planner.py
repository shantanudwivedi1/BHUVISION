import re
from collections.abc import Iterable

from backend.models.schemas import QueryPlan
from backend.planner.llm import LLMPlannerError, LLMQueryPlanner


_MODALITY_PATTERNS = (
    ("optical", r"\boptical\b"),
    ("sar", r"\bsar\b"),
    ("multispectral", r"\bmultispectral\b"),
    ("hyperspectral", r"\bhyperspectral\b"),
    ("thermal", r"\bthermal\b"),
    ("radar", r"\bradar\b"),
)

_OBJECT_PATTERNS = (
    ("buildings", r"\bbuildings?\b"),
    ("water", r"\bwater\b|\brivers?\b|\blakes?\b"),
    ("roads", r"\broads?\b|\bhighways?\b|\bstreets?\b"),
    ("ships", r"\bships?\b|\bvessels?\b"),
    ("vegetation", r"\bvegetation\b|\btrees?\b|\bforests?\b|\bcrops?\b|\bplants?\b"),
    ("vehicles", r"\bvehicles?\b|\bcars?\b|\btrucks?\b"),
    ("clouds", r"\bclouds?\b"),
)

_CHANGE_PATTERN = r"\b(change|changed|changes|compare|compared|between|appeared|disappeared)\b"


def build_query_plan(
    query: str,
    image_count: int = 0,
    modalities: Iterable[str] | None = None,
) -> QueryPlan:
    """Convert a query into a deterministic, validated QueryPlan."""
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    stated_modalities = _extract_modalities(normalized)
    supplied_modalities = list(modalities or [])
    effective_modalities = _merge_unique(stated_modalities, supplied_modalities)
    inferred_count = _infer_image_count(normalized, image_count)
    objects = _extract_objects(normalized)

    if "optical" in effective_modalities and "sar" in effective_modalities:
        return QueryPlan(
            task="optical_sar_analysis",
            objects=objects,
            image_count=max(inferred_count, 2),
            modalities=effective_modalities,
            required_evidence=_evidence_for(objects, "optical imagery", "SAR imagery"),
            output_types=["analysis", "evidence"],
        )

    if re.search(_CHANGE_PATTERN, normalized) and inferred_count >= 2:
        return QueryPlan(
            task="change_detection",
            objects=objects,
            image_count=inferred_count,
            temporal=True,
            modalities=effective_modalities,
            required_evidence=_evidence_for(objects, "image pair", "temporal difference"),
            output_types=["change_map", "evidence"],
        )

    if re.search(r"\b(describe|caption|summari[sz]e)\b", normalized):
        return QueryPlan(
            task="captioning",
            objects=objects,
            image_count=inferred_count,
            modalities=effective_modalities,
            required_evidence=["image content"],
            output_types=["caption"],
        )

    if re.search(r"\b(what|which|where|is there|identify|visible|detect|show|find|list)\b", normalized):
        output_types = ["object_list", "evidence"] if objects and re.search(r"\b(show|find|list)\b", normalized) else ["answer", "evidence"]
        return QueryPlan(
            task="vqa",
            objects=objects,
            image_count=inferred_count,
            modalities=effective_modalities,
            required_evidence=_evidence_for(objects, "image content", "question-relevant evidence"),
            output_types=output_types,
        )

    return QueryPlan(
        task="unsupported",
        objects=objects,
        image_count=image_count if image_count > 0 else 0,
        modalities=effective_modalities,
        required_evidence=[],
        output_types=[],
    )


def build_query_plan_with_optional_llm(
    query: str,
    image_count: int = 0,
    modalities: Iterable[str] | None = None,
    llm_planner: LLMQueryPlanner | None = None,
) -> QueryPlan:
    """Use the optional LLM for task classification, then enrich deterministically.

    Any LLM failure returns the deterministic plan. The LLM never supplies or
    changes modalities, objects, image counts, evidence, or output types.
    """
    deterministic_plan = build_query_plan(query, image_count, modalities)
    planner = llm_planner or LLMQueryPlanner()
    try:
        task = planner.plan_task(query)
    except LLMPlannerError:
        return deterministic_plan

    if task == "optical_sar_analysis" and not {"optical", "sar"}.issubset(deterministic_plan.modalities):
        return deterministic_plan
    if task == "change_detection" and deterministic_plan.image_count < 2:
        return deterministic_plan
    return _apply_task_metadata(deterministic_plan, task, query)


def _extract_modalities(query: str) -> list[str]:
    return [name for name, pattern in _MODALITY_PATTERNS if re.search(pattern, query)]


def _extract_objects(query: str) -> list[str]:
    return [name for name, pattern in _OBJECT_PATTERNS if re.search(pattern, query)]


def _infer_image_count(query: str, supplied_count: int) -> int:
    if supplied_count > 0:
        return supplied_count
    if re.search(r"\b(two|2|pair|both|multiple|several)\b", query):
        return 2
    match = re.search(r"\b(three|3|four|4|five|5)\b", query)
    if match:
        number_words = {"three": 3, "four": 4, "five": 5}
        return number_words.get(match.group(1), int(match.group(1))) if match.group(1).isdigit() else number_words[match.group(1)]
    if re.search(r"\b(one|1)\b", query):
        return 1
    return 1 if re.search(r"\b(image|images|imagery|scene)\b", query) else 0


def _merge_unique(first: Iterable[str], second: Iterable[str]) -> list[str]:
    values: list[str] = []
    for value in [*first, *second]:
        if value not in values:
            values.append(value)
    return values


def _apply_task_metadata(plan: QueryPlan, task: str, query: str) -> QueryPlan:
    updates: dict[str, object] = {"task": task}
    if task == "captioning":
        updates.update(required_evidence=["image content"], output_types=["caption"], temporal=False)
    elif task == "vqa":
        output_types = ["object_list", "evidence"] if plan.objects and re.search(r"\b(show|find|list)\b", query.lower()) else ["answer", "evidence"]
        updates.update(
            required_evidence=_evidence_for(plan.objects, "image content", "question-relevant evidence"),
            output_types=output_types,
            temporal=False,
        )
    elif task == "change_detection":
        updates.update(
            required_evidence=_evidence_for(plan.objects, "image pair", "temporal difference"),
            output_types=["change_map", "evidence"],
            temporal=True,
        )
    elif task == "optical_sar_analysis":
        updates.update(
            required_evidence=_evidence_for(plan.objects, "optical imagery", "SAR imagery"),
            output_types=["analysis", "evidence"],
            temporal=False,
        )
    else:
        updates.update(required_evidence=[], output_types=[], temporal=False)
    return plan.model_copy(update=updates)


def _evidence_for(objects: list[str], *required: str) -> list[str]:
    evidence = list(required)
    if objects:
        evidence.append("object evidence")
    return evidence

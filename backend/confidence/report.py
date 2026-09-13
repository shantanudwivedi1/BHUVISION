from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from backend.models.schemas import ConfidenceReport, ConfidenceSignal, Evidence


def build_confidence_report(
    evidence: Iterable[Evidence],
    image_analysis: Iterable[dict[str, Any]] | None = None,
) -> ConfidenceReport:
    evidence_list = list(evidence)
    signals = [
        _model_confidence_signal(evidence_list),
        _model_agreement_signal(evidence_list),
        _spatial_consistency_signal(evidence_list),
        _input_quality_signal(list(image_analysis or [])),
        ConfidenceSignal(
            name="paired_image_alignment",
            status="unavailable",
            explanation="No reliable geometric registration or alignment check is implemented.",
            metrics={"checked": False},
        ),
    ]
    scalar_values = [signal.value for signal in signals if signal.status == "available" and signal.value is not None]
    aggregate_value = round(sum(scalar_values) / len(scalar_values), 6) if scalar_values else None
    notes = []
    if aggregate_value is None:
        notes.append("No scalar confidence signals were available; no aggregate confidence was computed.")
    else:
        notes.append("Aggregate is the arithmetic mean of available scalar signals only.")
    if any(signal.status == "available" and signal.value is None for signal in signals):
        notes.append("Some input metrics were available without a defensible scalar confidence value.")
    return ConfidenceReport(
        signals=signals,
        aggregate_value=aggregate_value,
        aggregation_method="arithmetic_mean_of_available_scalar_signals" if scalar_values else "none",
        notes=notes,
    )


def _model_confidence_signal(evidence: list[Evidence]) -> ConfidenceSignal:
    values = [
        item.confidence
        for item in evidence
        if item.confidence is not None and _model_for(item) is not None
    ]
    if not values:
        return ConfidenceSignal(
            name="model_confidence",
            status="unavailable",
            explanation="No model-provided confidence values were supplied.",
        )
    return ConfidenceSignal(
        name="model_confidence",
        status="available",
        value=round(sum(values) / len(values), 6),
        metrics={"observation_count": len(values)},
        explanation="Mean of confidence values explicitly supplied by model evidence.",
        provenance={"models": sorted({_model_for(item) for item in evidence if item.confidence is not None and _model_for(item)})},
    )


def _model_agreement_signal(evidence: list[Evidence]) -> ConfidenceSignal:
    findings = [(_model_for(item), _normalise_finding(item.finding)) for item in evidence if _model_for(item) and item.finding]
    models = sorted({model for model, _ in findings})
    if len(models) < 2:
        return ConfidenceSignal(
            name="model_agreement",
            status="unavailable",
            explanation="Agreement requires findings from at least two distinct models.",
            metrics={"model_count": len(models)},
        )
    counts = Counter(finding for _, finding in findings)
    agreement = max(counts.values()) / len(findings)
    return ConfidenceSignal(
        name="model_agreement",
        status="available",
        value=round(agreement, 6),
        metrics={"model_count": len(models), "finding_count": len(findings), "finding_counts": dict(counts)},
        explanation="Share of model findings matching the most common normalized finding.",
        provenance={"models": models},
    )


def _spatial_consistency_signal(evidence: list[Evidence]) -> ConfidenceSignal:
    boxes = _spatial_boxes(evidence)
    if len(boxes) >= 2:
        values = [_box_iou(first[0], second[0]) for first, second in combinations(boxes, 2)]
        return ConfidenceSignal(
            name="spatial_consistency",
            status="available",
            value=round(sum(values) / len(values), 6),
            metrics={"method": "pairwise_bbox_iou", "comparison_count": len(values)},
            explanation="Mean pairwise intersection-over-union for supplied bounding boxes.",
            provenance={"coordinate_system": boxes[0][1] or "unspecified_shared_frame"},
        )

    masks = _spatial_masks(evidence)
    if len(masks) >= 2 and len({mask[1] for mask in masks}) == 1 and len({mask[2] for mask in masks}) == 1:
        values = [_mask_iou(first[0], second[0]) for first, second in combinations(masks, 2)]
        return ConfidenceSignal(
            name="spatial_consistency",
            status="available",
            value=round(sum(values) / len(values), 6),
            metrics={"method": "pairwise_mask_iou", "comparison_count": len(values), "shape": list(masks[0][1])},
            explanation="Mean pairwise intersection-over-union for supplied masks.",
        )

    coordinates = _spatial_coordinates(evidence)
    if len(coordinates) >= 2 and len({entry[1] for entry in coordinates}) == 1:
        values = [_coordinate_iou(first[0], second[0]) for first, second in combinations(coordinates, 2)]
        return ConfidenceSignal(
            name="spatial_consistency",
            status="available",
            value=round(sum(values) / len(values), 6),
            metrics={"method": "pairwise_coordinate_overlap", "comparison_count": len(values)},
            explanation="Mean pairwise overlap for supplied coordinates in a shared coordinate system.",
        )

    return ConfidenceSignal(
        name="spatial_consistency",
        status="unavailable",
        explanation="No comparable bounding boxes, masks, or coordinate sets were supplied.",
    )


def _input_quality_signal(image_analysis: list[dict[str, Any]]) -> ConfidenceSignal:
    if not image_analysis:
        return ConfidenceSignal(
            name="input_image_quality",
            status="unavailable",
            explanation="No inspected image quality metrics were supplied.",
        )
    qualities = [item.get("quality", {}) for item in image_analysis]
    dimensions = [(item.get("width"), item.get("height")) for item in image_analysis]
    return ConfidenceSignal(
        name="input_image_quality",
        status="available",
        metrics={
            "image_count": len(image_analysis),
            "formats": sorted({item.get("format") for item in image_analysis if item.get("format")}),
            "dimensions": dimensions,
            "total_file_size_bytes": sum(int(quality.get("file_size_bytes", 0)) for quality in qualities),
            "warning_count": sum(len(quality.get("warnings", [])) for quality in qualities),
        },
        explanation="Image dimensions, formats, file sizes, and inspection warnings were measured; no scalar confidence was assigned.",
        provenance={"method": "lightweight_image_header_inspection"},
    )


def _model_for(item: Evidence) -> str | None:
    return item.model or item.provenance.get("model")


def _normalise_finding(finding: str | None) -> str:
    return " ".join((finding or "").lower().split())


def _spatial_boxes(evidence: list[Evidence]) -> list[tuple[tuple[float, float, float, float], str]]:
    boxes = []
    for item in evidence:
        if not item.spatial or "bbox" not in item.spatial:
            continue
        bbox = item.spatial["bbox"]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                boxes.append((tuple(float(value) for value in bbox), str(item.spatial.get("coordinate_system", ""))))
            except (TypeError, ValueError):
                continue
    frames = Counter(frame for _, frame in boxes)
    frame, count = frames.most_common(1)[0] if frames else ("", 0)
    return [entry for entry in boxes if entry[1] == frame] if count >= 2 else []


def _box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return round(intersection / union, 6) if union else 0.0


def _spatial_masks(evidence: list[Evidence]) -> list[tuple[list[list[Any]], tuple[int, ...], str]]:
    masks = []
    for item in evidence:
        mask = item.spatial.get("mask") if item.spatial else None
        shape = _matrix_shape(mask)
        if shape:
            masks.append((mask, shape, str(item.spatial.get("coordinate_system", ""))))
    return masks


def _mask_iou(first: list[list[Any]], second: list[list[Any]]) -> float:
    first_values = [bool(value) for row in first for value in row]
    second_values = [bool(value) for row in second for value in row]
    intersection = sum(left and right for left, right in zip(first_values, second_values))
    union = sum(left or right for left, right in zip(first_values, second_values))
    return round(intersection / union, 6) if union else 0.0


def _spatial_coordinates(evidence: list[Evidence]) -> list[tuple[set[tuple[Any, ...]], str]]:
    coordinates = []
    for item in evidence:
        if not item.spatial or not isinstance(item.spatial.get("coordinates"), list):
            continue
        try:
            points = {tuple(point) for point in item.spatial["coordinates"] if isinstance(point, (list, tuple))}
            frame = str(item.spatial.get("coordinate_system", ""))
        except TypeError:
            continue
        if points and frame:
            coordinates.append((points, frame))
    return coordinates


def _coordinate_iou(first: set[tuple[Any, ...]], second: set[tuple[Any, ...]]) -> float:
    union = first | second
    return round(len(first & second) / len(union), 6) if union else 0.0


def _matrix_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        return None
    width = len(value[0])
    if not width or not all(len(row) == width for row in value):
        return None
    return len(value), width

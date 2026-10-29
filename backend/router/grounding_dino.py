"""Grounding DINO adapter for open-vocabulary, text-prompted object detection.

Uses the repository's local Grounding DINO Tiny weights for zero-shot object
detection and grounding using arbitrary natural language prompts.
Returns both detections (class, confidence, bounding boxes) and an annotated image.
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from backend.router.specialist import SpecialistDescriptor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundingDINOConfig:
    enabled: bool = True
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    device: str = "auto"

    @classmethod
    def from_env(cls) -> "GroundingDINOConfig":
        repo_model = Path(__file__).resolve().parents[2] / "models" / "grounding-dino-tiny"
        box_th = os.getenv("BHUVISION_GROUNDING_DINO_BOX_THRESHOLD",
                 os.getenv("BHUVISION_GROUNDING_DINO_CONFIDENCE", "0.25")).strip()
        text_th = os.getenv("BHUVISION_GROUNDING_DINO_TEXT_THRESHOLD", "0.25").strip()

        try:
            box_threshold = max(0.01, min(1.0, float(box_th)))
        except ValueError:
            box_threshold = 0.25

        try:
            text_threshold = max(0.01, min(1.0, float(text_th)))
        except ValueError:
            text_threshold = 0.25

        device = os.getenv("BHUVISION_GROUNDING_DINO_DEVICE", "auto").strip().lower()

        return cls(
            enabled=os.getenv("BHUVISION_GROUNDING_DINO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            model_id=os.getenv("BHUVISION_GROUNDING_DINO_MODEL", str(repo_model)).strip(),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )


# ---------------------------------------------------------------------------
# Prompt formatting for Grounding DINO
# ---------------------------------------------------------------------------

DEFAULT_REMOTE_SENSING_CLASSES = [
    "building",
    "road",
    "water body",
    "vegetation",
    "aircraft",
    "vessel",
    "vehicle",
    "runway",
]


def _prepare_grounding_text(query: str = "") -> str:
    """Format user query into lowercase dot-separated phrases required by Grounding DINO.

    Grounding DINO expects: "building. aircraft. road."
    """
    clean = (query or "").lower().strip()

    # Strip conversational / command boilerplate
    boilerplate = [
        "detect and localize", "detect", "locate", "localize", "find",
        "highlight", "mark", "identify",
        "in this satellite image", "in this image", "in satellite imagery",
        "with bounding boxes", "bounding boxes", "bounding box",
        "with coordinates", "coordinates",
        "please", "can you", "what is visible", "describe",
    ]
    for phrase in boilerplate:
        clean = clean.replace(phrase, "")

    clean = clean.strip(" .?:;,!")

    terms: list[str] = []
    if clean:
        parts = re.split(r",|\band\b|\bor\b", clean)
        for p in parts:
            p = p.strip(" .?:;,!")
            # Drop filler words
            p = re.sub(r"\b(all|the|a|an|some|any)\b", "", p).strip()
            if p and len(p) > 1:
                terms.append(p)

    if not terms:
        terms = list(DEFAULT_REMOTE_SENSING_CLASSES)

    # Grounding DINO requires a period at the end of each phrase
    return ". ".join(terms) + "."


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GroundingDINOAdapter:
    """Specialist adapter for Grounding DINO open-vocabulary object detection."""

    model_name = "Grounding DINO"
    capabilities = ("object_detection", "grounding", "open_vocabulary")
    required_modalities = ("optical",)

    def __init__(self, config: GroundingDINOConfig | None = None):
        self.config = config or GroundingDINOConfig.from_env()
        self._processor = None
        self._model = None
        self._device = None

    # -- Protocol method ---------------------------------------------------

    def describe(self) -> SpecialistDescriptor:
        available, reason = self._availability()
        return SpecialistDescriptor(
            model_name=self.model_name,
            capabilities=self.capabilities,
            required_modalities=self.required_modalities,
            available=available,
            reason=reason,
        )

    # -- Public inference method -------------------------------------------

    def detect(self, image_bytes: bytes, query: str = "") -> dict[str, Any]:
        """Run open-vocabulary grounded object detection using natural language prompt.

        Returns:
            detections: list of dicts with {class, confidence, bbox, box_pct}
            detection_count: number of detections
            model: model identifier
            annotated_image: data URL of the image with bounding boxes drawn
        """
        processor, model, device = self._load()
        image = _open_image(image_bytes)
        text_prompt = _prepare_grounding_text(query)

        import torch

        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process into pixel-coordinate boxes [xmin, ymin, xmax, ymax]
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[(image.height, image.width)],
        )

        detections: list[dict[str, Any]] = []
        if results and len(results) > 0:
            res = results[0]
            scores = res.get("scores", [])
            labels = res.get("labels", [])
            boxes = res.get("boxes", [])

            for score, label, box in zip(scores, labels, boxes):
                conf = round(float(score), 4)
                xmin, ymin, xmax, ymax = [round(float(v), 1) for v in box]

                # Ensure valid dimensions
                if xmax <= xmin or ymax <= ymin:
                    continue

                # Compute percentage coordinates [ymin_pct, xmin_pct, ymax_pct, xmax_pct]
                ymin_pct = round((ymin / image.height) * 100, 1)
                xmin_pct = round((xmin / image.width) * 100, 1)
                ymax_pct = round((ymax / image.height) * 100, 1)
                xmax_pct = round((xmax / image.width) * 100, 1)

                clean_label = label.strip()

                detections.append({
                    "class": clean_label,
                    "confidence": conf,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "box_pct": [ymin_pct, xmin_pct, ymax_pct, xmax_pct],
                })

        annotated_b64 = None
        if detections:
            annotated_b64 = _render_bounding_boxes(image, detections)

        return {
            "detections": detections,
            "detection_count": len(detections),
            "model": "Grounding DINO Tiny (local)",
            "annotated_image": annotated_b64,
            "text_prompt": text_prompt,
        }

    # -- Internals ---------------------------------------------------------

    def _availability(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "Grounding DINO is disabled via BHUVISION_GROUNDING_DINO_ENABLED."
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: F401
            from huggingface_hub import try_to_load_from_cache
            model_path = Path(self.config.model_id)
            if model_path.is_dir():
                has_config = (model_path / "config.json").is_file()
                has_weights = any((model_path / f).is_file() for f in (
                    "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"
                ))
            else:
                has_config = try_to_load_from_cache(self.config.model_id, "config.json")
                has_weights = any(try_to_load_from_cache(self.config.model_id, f) for f in (
                    "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"
                ))
            if not (has_config and has_weights):
                return False, f"Grounding DINO weights are not installed locally ({self.config.model_id}); offline mode will not download them."
        except (ImportError, OSError, ValueError) as exc:
            return False, f"Local Grounding DINO is unavailable: {exc}"
        return True, "Grounding DINO Tiny local weights are available for open-set detection."

    def _load(self):
        """Lazy-load the Grounding DINO processor and model."""
        if self._model is None:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device

            self._device = device
            self._processor = AutoProcessor.from_pretrained(self.config.model_id, local_files_only=True)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.model_id, local_files_only=True
            ).to(device)
            self._model.eval()

        return self._processor, self._model, self._device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_image(data: bytes):
    """Open raw bytes as an RGB PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _render_bounding_boxes(image, detections: list[dict[str, Any]]) -> str:
    """Render bounding boxes and labels on image and return base64 PNG data URL."""
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    palette = [
        "#00FF66",  # bright green
        "#FF3366",  # neon red
        "#33CCFF",  # sky cyan
        "#FFCC00",  # amber yellow
        "#CC33FF",  # magenta
        "#FF9933",  # orange
    ]

    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        xmin, ymin, xmax, ymax = det["bbox"]

        # Draw box outline
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Draw label banner
        label = det["class"]
        conf_pct = int(det["confidence"] * 100)
        label_text = f"{label} ({conf_pct}%)"
        banner_h = 16
        banner_w = len(label_text) * 7 + 6
        banner_top = max(0, ymin - banner_h)
        draw.rectangle([xmin, banner_top, min(width, xmin + banner_w), banner_top + banner_h], fill=color)
        draw.text((xmin + 3, banner_top + 2), label_text, fill="#000000")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

"""YOLOv8 adapter for real-time object detection.

Uses the ultralytics YOLOv8-nano model which is lightweight (~6 MB)
and runs comfortably on CPU.  The model auto-downloads on first use.
Returns both detections and an annotated image with bounding boxes.
"""

import base64
import io
import os
from dataclasses import dataclass
from typing import Any

from backend.router.specialist import SpecialistDescriptor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YOLOConfig:
    enabled: bool = True
    model_id: str = "yolov8n.pt"  # nano variant, fast on CPU
    confidence: float = 0.25
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "YOLOConfig":
        conf = os.getenv("BHUVISION_YOLO_CONFIDENCE", "0.25").strip()
        try:
            confidence = max(0.01, min(1.0, float(conf)))
        except ValueError:
            confidence = 0.25
        return cls(
            enabled=os.getenv("BHUVISION_YOLO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            model_id=os.getenv("BHUVISION_YOLO_MODEL", "yolov8n.pt").strip(),
            confidence=confidence,
            device=os.getenv("BHUVISION_YOLO_DEVICE", "cpu").strip(),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class YOLOAdapter:
    """Specialist adapter for YOLOv8 object detection."""

    model_name = "YOLOv8"
    capabilities = ("object_detection",)
    required_modalities = ("optical",)

    def __init__(self, config: YOLOConfig | None = None):
        self.config = config or YOLOConfig.from_env()
        self._model = None  # lazy-loaded

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

    def detect(self, image_bytes: bytes) -> dict[str, Any]:
        """Run object detection and return labelled bounding boxes + annotated image."""
        model = self._load()
        image = _open_image(image_bytes)
        results = model.predict(
            source=image,
            conf=self.config.confidence,
            device=self.config.device,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        annotated_b64: str | None = None

        for result in results:
            # Collect each detected object
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    # bbox as [x1, y1, x2, y2] pixel coordinates
                    "bbox": [round(float(v), 1) for v in box.xyxy[0]],
                })

            # Generate annotated image with bounding boxes drawn on it
            annotated_bgr = result.plot()  # numpy array (H, W, 3) in BGR
            annotated_b64 = _numpy_to_data_url(annotated_bgr)

        return {
            "detections": detections,
            "detection_count": len(detections),
            "model": self.config.model_id,
            "annotated_image": annotated_b64,
        }

    # -- Internals ---------------------------------------------------------

    def _availability(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "YOLO is disabled via BHUVISION_YOLO_ENABLED."
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False, "ultralytics package is not installed."
        return True, "YOLOv8 is available for object detection."

    def _load(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_id)
        return self._model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_image(data: bytes):
    """Open raw bytes as an RGB PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _numpy_to_data_url(bgr_array) -> str:
    """Convert a BGR numpy array (from ultralytics plot) to a base64 data URL."""
    from PIL import Image
    # ultralytics .plot() returns BGR; convert to RGB for PIL
    rgb_array = bgr_array[..., ::-1]
    img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

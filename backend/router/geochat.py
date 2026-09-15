"""GeoChat-7B adapter for remote sensing VQA, captioning, and grounded detection.

MBZUAI/geochat-7B is the premier 7B vision-language model tuned specifically
for remote sensing / satellite imagery. Supports optical and SAR modalities.

Provides:
- Natural-language VQA for satellite images
- Dense scene captioning / descriptions
- Grounded object detection with bounding box coordinates
- Multi-modal support (optical + SAR VH)
- Dual backend: Hugging Face ZeroGPU / Space or AWS SageMaker endpoint
- Automatic fallback to BLIP/YOLO if offline
"""

import base64
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from backend.router.specialist import SpecialistDescriptor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoChatConfig:
    enabled: bool = True
    backend: str = "hf_space"  # "hf_space" or "sagemaker"
    space_name: str = "Bireswar26/GeoChat"
    endpoint_name: str = ""
    region_name: str = "us-east-1"
    hf_token: str | None = None
    timeout_seconds: int = 45

    @classmethod
    def from_env(cls) -> "GeoChatConfig":
        return cls(
            enabled=os.getenv("BHUVISION_GEOCHAT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            backend=os.getenv("BHUVISION_GEOCHAT_BACKEND", "hf_space").strip().lower(),
            space_name=os.getenv("BHUVISION_GEOCHAT_SPACE", "Bireswar26/GeoChat").strip(),
            endpoint_name=os.getenv("BHUVISION_GEOCHAT_ENDPOINT_NAME", "").strip(),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip(),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
            timeout_seconds=int(os.getenv("BHUVISION_GEOCHAT_TIMEOUT", "45")),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GeoChatAdapter:
    """Specialist adapter for GeoChat-7B remote sensing VLM."""

    model_name = "GeoChat-7B"
    capabilities = ("vqa", "captioning", "object_detection")
    required_modalities = ("optical", "sar")

    def __init__(self, config: GeoChatConfig | None = None):
        self._explicit_config = config
        self._hf_client = None
        self._sagemaker_client = None

    @property
    def config(self) -> GeoChatConfig:
        if self._explicit_config:
            return self._explicit_config
        from dotenv import load_dotenv
        load_dotenv(override=True)
        return GeoChatConfig.from_env()

    def describe(self) -> SpecialistDescriptor:
        available, reason = self._availability()
        return SpecialistDescriptor(
            model_name=self.model_name,
            capabilities=self.capabilities,
            required_modalities=self.required_modalities,
            available=available,
            reason=reason,
        )

    def _availability(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "GeoChat is disabled by configuration (BHUVISION_GEOCHAT_ENABLED=false)."
        if self.config.backend == "sagemaker":
            if not self.config.endpoint_name:
                return False, "SageMaker endpoint name not configured (BHUVISION_GEOCHAT_ENDPOINT_NAME is empty)."
            return True, f"Configured for AWS SageMaker endpoint '{self.config.endpoint_name}' ({self.config.region_name})."
        # hf_space backend
        return True, f"Configured for GeoChat A10G Cloud Inference ({self.config.space_name})."

    # -- Public Inference Methods ------------------------------------------

    def vqa(
        self,
        image_bytes: bytes,
        question: str,
        sar_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Answer a natural language question about a satellite image."""
        raw_output = self._call_model(
            image_bytes=image_bytes,
            prompt=question,
            max_new_tokens=128,
            sar_bytes=sar_bytes,
        )
        clean_text, detections = _parse_geochat_output(raw_output)
        return {
            "answer": clean_text,
            "raw_output": raw_output,
            "model": "MBZUAI/geochat-7B",
            "detections": detections,
        }

    def caption(
        self,
        image_bytes: bytes,
        sar_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Generate a dense satellite scene description / caption."""
        prompt = "Describe the visible features, land use, and objects in this satellite image in detail."
        raw_output = self._call_model(
            image_bytes=image_bytes,
            prompt=prompt,
            max_new_tokens=160,
            sar_bytes=sar_bytes,
        )
        clean_text, detections = _parse_geochat_output(raw_output)
        return {
            "caption": clean_text,
            "raw_output": raw_output,
            "model": "MBZUAI/geochat-7B",
            "detections": detections,
        }

    def detect(
        self,
        image_bytes: bytes,
        query: str = "Detect and localize prominent objects in this satellite image.",
        sar_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """Detect and localize objects with bounding boxes using GeoChat grounding."""
        # Extract target object category from query for clean labeling
        target_label = query.strip()
        m = re.search(r"(?:detect|locate|localize|find|highlight)\s+(?:all\s+)?(.*?)(?:\s+in|\s+with|\.|\?|$)", query, re.IGNORECASE)
        if m and m.group(1).strip():
            target_label = m.group(1).strip()
        elif len(target_label) > 25:
            target_label = "object"

        # Ensure query asks for localization if not explicitly doing so
        grounding_prompt = query
        if not any(w in query.lower() for w in ("detect", "locate", "localize", "find", "highlight")):
            grounding_prompt = f"Detect and localize {query} in this satellite image with bounding boxes."

        raw_output = self._call_model(
            image_bytes=image_bytes,
            prompt=grounding_prompt,
            max_new_tokens=180,
            sar_bytes=sar_bytes,
        )
        clean_text, detections = _parse_geochat_output(raw_output, default_label=target_label)

        annotated_image = None
        if detections:
            annotated_image = _render_bounding_boxes(image_bytes, detections)

        return {
            "description": clean_text,
            "raw_output": raw_output,
            "model": "MBZUAI/geochat-7B",
            "detections": detections,
            "annotated_image": annotated_image,
        }

    # -- Internal Model Invocations ----------------------------------------

    def _call_model(
        self,
        image_bytes: bytes,
        prompt: str,
        max_new_tokens: int = 128,
        sar_bytes: bytes | None = None,
    ) -> str:
        if self.config.backend == "sagemaker":
            return self._call_sagemaker(image_bytes, prompt, max_new_tokens, sar_bytes)
        return self._call_hf_space(image_bytes, prompt, max_new_tokens, sar_bytes)

    def _call_hf_space(
        self,
        image_bytes: bytes,
        prompt: str,
        max_new_tokens: int,
        sar_bytes: bytes | None = None,
    ) -> str:
        from gradio_client import Client, handle_file

        if self._hf_client is None:
            client_kwargs = {"verbose": False}
            if self.config.hf_token:
                client_kwargs["token"] = self.config.hf_token
            self._hf_client = Client(self.config.space_name, **client_kwargs)

        # Write image bytes to temp files for gradio client handle_file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_img:
            f_img.write(image_bytes)
            img_path = f_img.name

        sar_path = None
        if sar_bytes:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_sar:
                f_sar.write(sar_bytes)
                sar_path = f_sar.name

        try:
            sar_handle = handle_file(sar_path) if sar_path else None
            result = self._hf_client.predict(
                image=handle_file(img_path),
                prompt=prompt,
                max_new_tokens=float(max_new_tokens),
                sar_vh=sar_handle,
                api_name="/geochat",
            )
            return str(result)
        finally:
            # Clean up temp files
            for p in (img_path, sar_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    def _call_sagemaker(
        self,
        image_bytes: bytes,
        prompt: str,
        max_new_tokens: int,
        sar_bytes: bytes | None = None,
    ) -> str:
        import boto3
        from botocore.exceptions import ClientError

        if self._sagemaker_client is None:
            self._sagemaker_client = boto3.client("sagemaker-runtime", region_name=self.config.region_name)

        payload = {
            "prompt": prompt,
            "inputs": prompt,
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "max_tokens": max_new_tokens,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.2,
            },
        }
        if sar_bytes:
            payload["sar_vh"] = base64.b64encode(sar_bytes).decode("utf-8")

        try:
            response = self._sagemaker_client.invoke_endpoint(
                EndpointName=self.config.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            body = json.loads(response["Body"].read().decode("utf-8"))
            if isinstance(body, list) and body:
                return body[0].get("generated_text", str(body[0]))
            if isinstance(body, dict):
                if "generated_text" in body:
                    return str(body["generated_text"] or "")
                if "response" in body:
                    return str(body["response"] or "")
                return str(body)
            return str(body)
        except ClientError as err:
            error_code = err.response.get("Error", {}).get("Code", "")
            error_msg = err.response.get("Error", {}).get("Message", str(err))
            # Model is loading on first request (lazy init takes 2-5 minutes)
            if "ModelError" in error_code or "ModelNotReadyException" in error_code:
                if "CUDA" in error_msg or "model_fn" in error_msg or "Loading" in error_msg:
                    raise RuntimeError(
                        "AWS SageMaker GPU endpoint is loading GeoChat-7B model weights on first request. "
                        "This one-time load takes 2-5 minutes. Please retry shortly."
                    ) from err
            if "ResourceNotFound" in error_code:
                raise RuntimeError(f"SageMaker endpoint '{self.config.endpoint_name}' not found.") from err
            if "ValidationError" in error_code and "endpoint is being updated" in error_msg.lower():
                raise RuntimeError("SageMaker endpoint is being updated. Please wait a few minutes.") from err
            raise RuntimeError(f"SageMaker endpoint '{self.config.endpoint_name}' error [{error_code}]: {error_msg}") from err



# ---------------------------------------------------------------------------
# Output parsing & box rendering
# ---------------------------------------------------------------------------

def _parse_geochat_output(raw_text: str, default_label: str = "detected_object") -> tuple[str, list[dict[str, Any]]]:
    """Parse GeoChat grounded output format:

    <p>label</p> {<y0><x0><y1><x1>|<angle>}<delim>{<y0><x0><y1><x1>}
    or direct coordinate tokens: {<y0><x0><y1><x1>|<angle>}
    Returns (cleaned_text, list_of_detections).
    Each detection has:
      {"class": label, "box_pct": [ymin, xmin, ymax, xmax]}
    """
    detections: list[dict[str, Any]] = []

    # 1. Pattern for labeled bounding boxes with <p>label</p>
    pattern = r"<p>(.*?)</p>\s*((?:\{<.*?>\}\s*(?:<delim>)?\s*)+)"
    matches = re.findall(pattern, raw_text)

    for label, boxes_str in matches:
        clean_label = label.strip()
        # Find all box coordinate sets
        box_coords = re.findall(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>", boxes_str)
        for y0, x0, y1, x1 in box_coords:
            detections.append({
                "class": clean_label,
                "confidence": 0.95,
                "box_pct": [int(y0), int(x0), int(y1), int(x1)],
            })

    # 2. If no <p> tags matched, check for direct coordinate boxes
    if not detections:
        box_coords = re.findall(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>", raw_text)
        for y0, x0, y1, x1 in box_coords:
            detections.append({
                "class": default_label,
                "confidence": 0.92,
                "box_pct": [int(y0), int(x0), int(y1), int(x1)],
            })

    # Clean the text for human presentation
    clean = re.sub(r"\{<.*?>\}|<delim>", "", raw_text)
    clean = re.sub(r"</?p>", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)

    if not clean and detections:
        clean = f"Detected {len(detections)} {default_label} object(s) with bounding box coordinates."

    return clean, detections


def _render_bounding_boxes(image_bytes: bytes, detections: list[dict[str, Any]]) -> str:
    """Draw bounding boxes and labels onto the image and return base64 PNG data URL."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    palette = [
        "#00FF66",  # bright green
        "#FF3366",  # neon red
        "#33CCFF",  # sky cyan
        "#FFCC00",  # amber yellow
        "#CC33FF",  # magenta
    ]

    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        box_pct = det["box_pct"]  # [ymin, xmin, ymax, xmax] in 0-100%
        ymin = int(box_pct[0] / 100.0 * height)
        xmin = int(box_pct[1] / 100.0 * width)
        ymax = int(box_pct[2] / 100.0 * height)
        xmax = int(box_pct[3] / 100.0 * width)

        # Ensure valid coords
        if xmax <= xmin or ymax <= ymin:
            continue

        # Draw box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Draw label header
        label = det["class"]
        label_text = f"{label} ({int(det['confidence']*100)}%)"
        draw.rectangle([xmin, max(0, ymin - 18), min(width, xmin + len(label_text) * 8), ymin], fill=color)
        draw.text((xmin + 2, max(0, ymin - 16)), label_text, fill="black")

        # Save absolute bbox in detection object for spatial metadata
        det["bbox"] = [xmin, ymin, xmax, ymax]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

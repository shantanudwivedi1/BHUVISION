"""BLIP adapter for VQA and image captioning.

Uses the repository's local BLIP VQA weights for visual question answering
and image descriptions. Missing files are never downloaded automatically.
"""

import io
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from backend.router.specialist import SpecialistDescriptor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BLIPConfig:
    enabled: bool = True
    device: str = "cpu"
    vqa_model_id: str = "Salesforce/blip-vqa-base"
    caption_model_id: str = "Salesforce/blip-vqa-base"

    @classmethod
    def from_env(cls) -> "BLIPConfig":
        repo_models = Path(__file__).resolve().parents[2] / "models"
        return cls(
            enabled=os.getenv("BHUVISION_BLIP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            device=os.getenv("BHUVISION_BLIP_DEVICE", "cpu").strip(),
            vqa_model_id=os.getenv("BHUVISION_BLIP_VQA_MODEL", str(repo_models / "blip-vqa-base")).strip(),
            caption_model_id=os.getenv("BHUVISION_BLIP_CAPTION_MODEL", str(repo_models / "blip-vqa-base")).strip(),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BLIPAdapter:
    """Specialist adapter for BLIP VQA and captioning."""

    model_name = "BLIP"
    capabilities = ("vqa", "captioning")
    required_modalities = ("optical",)

    def __init__(self, config: BLIPConfig | None = None):
        self.config = config or BLIPConfig.from_env()
        # Lazy-loaded on first inference (never during describe / startup).
        self._vqa_model = None
        self._vqa_processor = None
        self._caption_model = None
        self._caption_processor = None

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

    # -- Public inference methods ------------------------------------------

    def vqa(self, image_bytes: bytes, question: str) -> dict[str, Any]:
        """Answer a natural-language question about an image."""
        model, processor = self._load_vqa()
        image = _open_image(image_bytes)
        inputs = processor(image, question, return_tensors="pt").to(self.config.device)
        out = model.generate(**inputs, max_new_tokens=64)
        answer = processor.decode(out[0], skip_special_tokens=True).strip()
        return {"answer": answer, "model": "Salesforce/blip-vqa-base (local)"}

    def caption(self, image_bytes: bytes) -> dict[str, Any]:
        """Generate a description / caption for an image.

        Uses the cached BLIP model (Salesforce/blip-vqa-base) which is
        already downloaded on disk, eliminating any 990 MB download on restart.
        """
        if self.config.caption_model_id != self.config.vqa_model_id and _is_model_cached(self.config.caption_model_id):
            model, processor = self._load_caption()
            image = _open_image(image_bytes)
            inputs = processor(image, return_tensors="pt").to(self.config.device)
            out = model.generate(**inputs, max_new_tokens=64)
            text = processor.decode(out[0], skip_special_tokens=True).strip()
            return {"caption": text, "model": "Salesforce/blip-vqa-base (local)"}

        # Seamlessly use the local cached VQA model
        vqa_res = self.vqa(image_bytes, "Describe what you see in this image")
        return {"caption": vqa_res["answer"], "model": vqa_res["model"]}

    # -- Internals ---------------------------------------------------------

    def _availability(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "BLIP is disabled via BHUVISION_BLIP_ENABLED."
        # Check that transformers + torch are importable.
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"BLIP runtime dependency missing: {exc}"
        if not _is_model_cached(self.config.vqa_model_id):
            return False, f"BLIP weights are not installed locally ({self.config.vqa_model_id}); offline mode will not download them."
        return True, "BLIP is available for VQA and captioning."

    def _load_vqa(self):
        """Lazy-load the VQA model and processor from cache."""
        if self._vqa_model is None:
            if not _is_model_cached(self.config.vqa_model_id):
                raise RuntimeError(
                    f"BLIP VQA model weights ({self.config.vqa_model_id}) are not installed locally. Offline mode will not download them."
                )
            from transformers import BlipProcessor, BlipForQuestionAnswering
            self._vqa_processor = BlipProcessor.from_pretrained(self.config.vqa_model_id, local_files_only=True)
            self._vqa_model = BlipForQuestionAnswering.from_pretrained(
                self.config.vqa_model_id,
                local_files_only=True,
            ).to(self.config.device).eval()
        return self._vqa_model, self._vqa_processor

    def _load_caption(self):
        """Lazy-load the captioning model and processor from cache."""
        if self._caption_model is None:
            if not _is_model_cached(self.config.caption_model_id):
                raise RuntimeError(
                    f"BLIP captioning model weights ({self.config.caption_model_id}) are not installed locally. Offline mode will not download them."
                )
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self._caption_processor = BlipProcessor.from_pretrained(self.config.caption_model_id, local_files_only=True)
            self._caption_model = BlipForConditionalGeneration.from_pretrained(
                self.config.caption_model_id,
                local_files_only=True,
            ).to(self.config.device).eval()
        return self._caption_model, self._caption_processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_image(data: bytes):
    """Open raw bytes as an RGB PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _is_model_cached(model_id: str) -> bool:
    """Check if model weights are present in local HuggingFace cache."""
    local_model = Path(model_id)
    if local_model.is_dir():
        return (local_model / "config.json").is_file() and any(
            (local_model / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
        )
    try:
        from huggingface_hub import try_to_load_from_cache
        for filename in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
            res = try_to_load_from_cache(model_id, filename)
            if res is not None and not str(res).endswith(".incomplete") and "incomplete" not in str(res):
                return True
        return False
    except Exception:
        return False

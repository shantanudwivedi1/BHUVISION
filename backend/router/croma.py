import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.router.specialist import SpecialistDescriptor


class CROMAPreprocessingError(ValueError):
    """Raised when Sentinel-1/2 arrays do not match CROMA's input contract."""


@dataclass(frozen=True)
class CROMAInputs:
    sar_images: Any
    optical_images: Any
    sar_shape: tuple[int, ...]
    optical_shape: tuple[int, ...]


class CROMAResult(BaseModel):
    status: Literal["features_ready", "capability_unavailable"]
    model_name: str = "CROMA"
    capabilities: list[str] = Field(default_factory=lambda: ["joint_representation", "feature_extraction"])
    required_modalities: list[str] = Field(default_factory=lambda: ["optical", "sar"])
    representation_type: str | None = None
    feature_shape: list[int] | None = None
    semantic_labels: list[str] = Field(default_factory=list)
    explanation: str


@dataclass(frozen=True)
class CROMAConfig:
    enabled: bool = False
    model_id: str = "BiliSakura/CROMA-transformers"
    model_path: str = ""
    device: str = "cuda"
    allow_download: bool = False
    allow_cpu: bool = False

    @classmethod
    def from_env(cls) -> "CROMAConfig":
        return cls(
            enabled=_as_bool(os.getenv("BHUVISION_CROMA_ENABLED", "false")),
            model_id=os.getenv("BHUVISION_CROMA_MODEL_ID", "BiliSakura/CROMA-transformers").strip(),
            model_path=os.getenv("BHUVISION_CROMA_MODEL_PATH", "").strip(),
            device=os.getenv("BHUVISION_CROMA_DEVICE", "cuda").strip(),
            allow_download=_as_bool(os.getenv("BHUVISION_CROMA_ALLOW_DOWNLOAD", "false")),
            allow_cpu=_as_bool(os.getenv("BHUVISION_CROMA_ALLOW_CPU", "false")),
        )


class CROMAAdapter:
    """Optional Transformers adapter for CROMA joint feature extraction."""

    model_name = "CROMA"
    capabilities = ("joint_representation", "feature_extraction")
    required_modalities = ("optical", "sar")

    def __init__(self, config: CROMAConfig | None = None):
        self.config = config or CROMAConfig.from_env()
        self._model = None
        self._processor = None

    def describe(self) -> SpecialistDescriptor:
        available, reason = self._availability()
        return SpecialistDescriptor(
            model_name=self.model_name,
            capabilities=self.capabilities,
            required_modalities=self.required_modalities,
            available=available,
            reason=reason,
        )

    def run(self, sar_images: Any, optical_images: Any) -> CROMAResult:
        descriptor = self.describe()
        if not descriptor.available:
            return CROMAResult(
                status="capability_unavailable",
                explanation=descriptor.reason,
            )

        try:
            inputs = prepare_croma_inputs(sar_images, optical_images)
            model, processor, torch = self._load_runtime()
            batch = processor(
                sar_images=inputs.sar_images,
                optical_images=inputs.optical_images,
                return_tensors="pt",
            )
            batch = {
                key: value.to(self.config.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            with torch.inference_mode():
                outputs = model(**batch)
            pooled = outputs.get("pooler_output") if hasattr(outputs, "get") else getattr(outputs, "pooler_output", None)
            if pooled is None:
                raise RuntimeError("CROMA output did not contain pooler_output")
            shape = [int(value) for value in pooled.shape]
            return CROMAResult(
                status="features_ready",
                representation_type="joint_embedding",
                feature_shape=shape,
                explanation=(
                    "CROMA produced a learned joint optical-SAR representation. "
                    "No semantic labels or confidence were inferred."
                ),
            )
        except Exception as exc:
            return CROMAResult(
                status="capability_unavailable",
                explanation=f"CROMA could not run: {type(exc).__name__}. No semantic result was produced.",
            )

    def _availability(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "CROMA is disabled. Enable it only in a GPU-capable runtime."
        if not self.config.model_id and not self.config.model_path:
            return False, "No CROMA model ID or local model path is configured."
        if not self.config.allow_download and not self.config.model_path:
            return False, (
                "CROMA model download is disabled and no local model path is configured. "
                "Set BHUVISION_CROMA_MODEL_PATH or explicitly allow download on a GPU host."
            )
        missing = [name for name in ("torch", "transformers", "einops", "numpy") if importlib.util.find_spec(name) is None]
        if missing:
            return False, f"CROMA runtime dependencies are missing: {', '.join(missing)}."
        if self.config.device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                return False, "CROMA requires a CUDA GPU for the configured device, but CUDA is unavailable."
        elif not self.config.allow_cpu:
            return False, "CPU execution is disabled for CROMA; use a CUDA GPU or set BHUVISION_CROMA_ALLOW_CPU=true explicitly."
        if self.config.model_path and not Path(self.config.model_path).exists():
            return False, f"Configured CROMA model path does not exist: {self.config.model_path}."
        return True, "CROMA runtime and model source are available."

    def _load_runtime(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        source = self.config.model_path or self.config.model_id
        local_only = not self.config.allow_download
        self._processor = self._processor or AutoImageProcessor.from_pretrained(
            source, trust_remote_code=True, local_files_only=local_only
        )
        self._model = self._model or AutoModel.from_pretrained(
            source, trust_remote_code=True, local_files_only=local_only
        ).to(self.config.device).eval()
        return self._model, self._processor, torch


def prepare_croma_inputs(sar_images: Any, optical_images: Any) -> CROMAInputs:
    """Validate Sentinel-1/2 channel contracts without importing torch or GPU code.

    CROMA expects SAR with 2 channels and optical with 12 channels. A 13-band
    Sentinel-2 input is accepted and has B10 (the cirrus band, index 10 in the
    conventional B1..B12/B8A ordering) removed.
    """
    sar_shape = _shape_of(sar_images, "SAR")
    optical_shape = _shape_of(optical_images, "optical")
    sar_channel_axis = _channel_axis(sar_shape, "SAR")
    optical_channel_axis = _channel_axis(optical_shape, "optical")
    sar_channels = sar_shape[sar_channel_axis]
    optical_channels = optical_shape[optical_channel_axis]
    if sar_channels != 2:
        raise CROMAPreprocessingError(f"Sentinel-1 SAR must have 2 channels; received {sar_channels}.")
    if optical_channels == 13:
        optical_images = _remove_channel(optical_images, optical_channel_axis, 10)
        optical_shape = _shape_of(optical_images, "optical")
        optical_channels = optical_shape[optical_channel_axis]
    if optical_channels != 12:
        raise CROMAPreprocessingError(
            f"Sentinel-2 optical must have 12 channels after B10 removal; received {optical_channels}."
        )
    if sar_shape[-2:] != optical_shape[-2:]:
        raise CROMAPreprocessingError(
            f"SAR and optical spatial sizes must match; received {sar_shape[-2:]} and {optical_shape[-2:]}.")
    return CROMAInputs(sar_images, optical_images, sar_shape, optical_shape)


def _shape_of(value: Any, name: str) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        shape = tuple(int(size) for size in value.shape)
    else:
        shape = _nested_shape(value)
    if len(shape) not in (3, 4):
        raise CROMAPreprocessingError(f"{name} input must have shape (C,H,W) or (B,C,H,W); received {shape}.")
    return shape


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        return ()
    child_shape = _nested_shape(value[0])
    if any(_nested_shape(child) != child_shape for child in value):
        raise CROMAPreprocessingError("Input arrays must be rectangular.")
    return (len(value), *child_shape)


def _channel_axis(shape: tuple[int, ...], name: str) -> int:
    axis = 1 if len(shape) == 4 else 0
    if shape[-2] <= 0 or shape[-1] <= 0:
        raise CROMAPreprocessingError(f"{name} input has invalid spatial dimensions: {shape}.")
    return axis


def _remove_channel(value: Any, axis: int, index: int) -> Any:
    if hasattr(value, "shape"):
        try:
            import numpy as np

            return np.delete(value, index, axis=axis)
        except ImportError:
            pass
    if axis == 0:
        return value[:index] + value[index + 1:]
    return [sample[:index] + sample[index + 1:] for sample in value]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}

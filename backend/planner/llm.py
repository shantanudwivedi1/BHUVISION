import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.models.schemas import TaskName


class LLMPlannerError(RuntimeError):
    """Raised when the optional LLM planner cannot produce a valid task."""


class LLMTaskResponse(BaseModel):
    """The only field the LLM is allowed to determine."""

    model_config = ConfigDict(extra="forbid")

    task: TaskName


@dataclass(frozen=True)
class LLMPlannerConfig:
    provider: str = "none"
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "LLMPlannerConfig":
        timeout = os.getenv("BHUVISION_LLM_TIMEOUT_SECONDS", "10")
        try:
            timeout_seconds = max(1.0, float(timeout))
        except ValueError:
            timeout_seconds = 10.0
        return cls(
            provider=os.getenv("BHUVISION_LLM_PROVIDER", "none").strip().lower(),
            api_key=os.getenv("BHUVISION_LLM_API_KEY", "").strip(),
            model=os.getenv("BHUVISION_LLM_MODEL", "").strip(),
            base_url=os.getenv("BHUVISION_LLM_BASE_URL", "").strip(),
            timeout_seconds=timeout_seconds,
        )


class LLMQueryPlanner:
    """Small provider adapter for an OpenAI-compatible chat-completions API."""

    _TASKS = ["vqa", "captioning", "change_detection", "optical_sar_analysis", "unsupported"]

    def __init__(self, config: LLMPlannerConfig | None = None):
        self.config = config or LLMPlannerConfig.from_env()

    def plan_task(self, query: str) -> TaskName:
        if self.config.provider in {"", "none", "disabled"}:
            raise LLMPlannerError("LLM planner is disabled")
        if self.config.provider != "openai_compatible":
            raise LLMPlannerError(f"Unsupported LLM provider: {self.config.provider}")
        if not self.config.base_url or not self.config.model or not self.config.api_key:
            raise LLMPlannerError("LLM provider is missing base URL, model, or API key")

        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify the user's satellite query into exactly one task. "
                        "Return JSON only with one key, task. Do not choose models, modalities, "
                        "objects, evidence, or output types. Allowed tasks: "
                        + ", ".join(self._TASKS)
                    ),
                },
                {"role": "user", "content": query},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "bhuvision_query_task",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"task": {"type": "string", "enum": self._TASKS}},
                        "required": ["task"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        request = Request(
            url=self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMPlannerError("LLM request failed") from exc

        content = _extract_content(response_payload)
        try:
            parsed = LLMTaskResponse.model_validate_json(_strip_code_fence(content))
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMPlannerError("LLM returned an invalid structured task") from exc
        return parsed.task


def _extract_content(response_payload: dict[str, Any]) -> str:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMPlannerError("LLM response did not contain message content") from exc
    if not isinstance(content, str):
        raise LLMPlannerError("LLM response content was not text")
    return content.strip()


def _strip_code_fence(content: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else content

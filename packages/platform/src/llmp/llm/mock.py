"""Deterministic LLM client for tests and CI.

- `schema` mode returns a fixed, schema-valid instance of the requested output model. It exercises
  plumbing only and must never be used to produce quality metrics.
- `replay` mode returns responses previously recorded from a real backend ("cassettes"), keyed by
  a hash of everything that influences generation. A missing cassette is an error, never a guess.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from llmp.llm.base import LLMError
from llmp.llm.types import LLMRequest, LLMResponse, Usage

MOCK_TEXT = "mock response"


class CassetteMissError(LLMError):
    """No recorded response exists for this request."""


def cassette_key(request: LLMRequest, model: str) -> str:
    """Stable hash of every request field that influences the generated output."""
    payload = {
        "model": model,
        "messages": [[m.role, m.content] for m in request.messages],
        "schema": request.output_model.model_json_schema() if request.output_model else None,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class CassetteStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def path(self, key: str) -> Path:
        return self._root / f"{key}.json"

    def load(self, key: str) -> LLMResponse:
        path = self.path(key)
        if not path.exists():
            raise CassetteMissError(
                f"No cassette {path}. The request changed (prompt, schema or params) since the "
                "last recording: re-record against a real vLLM."
            )
        return LLMResponse.model_validate(json.loads(path.read_text(encoding="utf-8"))["response"])

    def save(self, key: str, request: LLMRequest, response: LLMResponse) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        record = {"prompt": request.prompt.model_dump(), "response": response.model_dump()}
        self.path(key).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


class MockLLMClient:
    def __init__(
        self, mode: Literal["schema", "replay"], model: str, cassettes: CassetteStore | None = None
    ) -> None:
        if mode == "replay" and cassettes is None:
            raise ValueError("replay mode requires a CassetteStore")
        self._mode = mode
        self._model = model
        self._cassettes = cassettes

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._cassettes is not None and self._mode == "replay":
            return self._cassettes.load(cassette_key(request, self._model))
        if request.output_model is None:
            text = MOCK_TEXT
        else:
            schema = request.output_model.model_json_schema()
            instance = _instance(schema, schema.get("$defs", {}))
            text = request.output_model.model_validate(instance).model_dump_json()
        return LLMResponse(
            text=text,
            model=self._model,
            usage=Usage(prompt_tokens=0, completion_tokens=0),
            latency_s=0.0,
        )

    async def ready(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


def _instance(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Build a minimal deterministic value satisfying a pydantic-generated JSON schema."""
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        options = [s for s in schema["anyOf"] if s.get("type") != "null"] or schema["anyOf"]
        return _instance(options[0], defs)

    match schema.get("type"):
        case "object":
            props = schema.get("properties", {})
            return {name: _instance(sub, defs) for name, sub in props.items()}
        case "array":
            return [_instance(schema["items"], defs)] * max(schema.get("minItems", 1), 1)
        case "string":
            if "pattern" in schema:
                raise ValueError(f"schema mock does not support string patterns: {schema}")
            fmt = schema.get("format")
            if fmt == "date":
                return "2026-01-01"
            if fmt == "date-time":
                return "2026-01-01T00:00:00Z"
            return "x" * max(schema.get("minLength", 1), 1)
        case "integer" | "number":
            return _lower_bound(schema)
        case "boolean":
            return False
        case "null":
            return None
    raise ValueError(f"schema mock does not support: {schema}")


def _lower_bound(schema: dict[str, Any]) -> int | float:
    if "minimum" in schema:
        return schema["minimum"]  # type: ignore[no-any-return]
    if "exclusiveMinimum" in schema:
        return schema["exclusiveMinimum"] + 1  # type: ignore[no-any-return]
    return 0

import time
from typing import Any

import httpx

from llmp.llm.base import LLMError
from llmp.llm.types import LLMRequest, LLMResponse, Usage


class VLLMClient:
    """Client for vLLM's OpenAI-compatible chat completions endpoint."""

    def __init__(self, http: httpx.AsyncClient, model: str) -> None:
        self._http = http
        self._model = model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.output_model is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_model.__name__,
                    "schema": request.output_model.model_json_schema(),
                    "strict": True,
                },
            }

        start = time.perf_counter()
        try:
            resp = await self._http.post("/v1/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"vLLM returned {e.response.status_code}: {e.response.text}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"vLLM request failed: {e!r}") from e
        latency = time.perf_counter() - start

        data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
            ),
            latency_s=latency,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

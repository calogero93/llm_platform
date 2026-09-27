import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from llmp.llm import LLMError, LLMRequest, Message, PromptRef
from llmp.llm.vllm import VLLMClient


class Answer(BaseModel):
    value: int


REQUEST = LLMRequest(
    messages=(Message("system", "be terse"), Message("user", "2+2?")),
    prompt=PromptRef("math", 1, "abc"),
    output_model=Answer,
    max_tokens=16,
)

COMPLETION = {
    "model": "qwen",
    "choices": [{"message": {"role": "assistant", "content": '{"value": 4}'}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
}


def client_with(handler: Any) -> VLLMClient:
    http = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(handler))
    return VLLMClient(http, model="qwen")


async def test_sends_openai_request_with_json_schema() -> None:
    sent: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent.update(path=req.url.path, body=json.loads(req.content))
        return httpx.Response(200, json=COMPLETION)

    await client_with(handler).complete(REQUEST)

    assert sent["path"] == "/v1/chat/completions"
    body = sent["body"]
    assert body["model"] == "qwen"
    assert body["messages"][1] == {"role": "user", "content": "2+2?"}
    assert body["max_tokens"] == 16
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()


async def test_parses_response_and_usage() -> None:
    resp = await client_with(lambda _: httpx.Response(200, json=COMPLETION)).complete(REQUEST)
    assert resp.parse(Answer).value == 4
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (12, 5)
    assert resp.model == "qwen"
    assert resp.latency_s >= 0


async def test_plain_text_request_has_no_response_format() -> None:
    sent: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent.update(json.loads(req.content))
        return httpx.Response(200, json=COMPLETION)

    plain = LLMRequest(messages=REQUEST.messages, prompt=REQUEST.prompt)
    await client_with(handler).complete(plain)
    assert "response_format" not in sent


async def test_server_error_raises_llm_error() -> None:
    client = client_with(lambda _: httpx.Response(503, text="loading"))
    with pytest.raises(LLMError, match="503"):
        await client.complete(REQUEST)


async def test_timeout_raises_llm_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(LLMError, match="ReadTimeout"):
        await client_with(handler).complete(REQUEST)

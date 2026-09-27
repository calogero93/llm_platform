import json

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import llmp.api.app
from llmp.api.app import create_app
from llmp.config import Settings
from llmp.eval import EvalSuite
from llmp.llm import LLMRequest, Message, PromptRef
from llmp.llm.mock import MockLLMClient
from llmp.logs import JsonFormatter
from llmp.modules import Module, PlatformContext
from llmp.tracing import TracedLLMClient

REQUEST = LLMRequest(
    messages=(Message(role="user", content="hello"),),
    prompt=PromptRef(id="greet", version=2, hash="abc123"),
    max_tokens=8,
)


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(llmp.api.app, "build_tracer_provider", lambda _: provider)
    return exporter


def by_name(spans: tuple[ReadableSpan, ...], name: str) -> ReadableSpan:
    (span,) = [s for s in spans if s.name == name]
    return span


async def test_llm_span_records_prompt_model_tokens_and_io() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = TracedLLMClient(MockLLMClient("schema", model="m"), provider.get_tracer("t"))

    await client.complete(REQUEST)

    attrs = by_name(exporter.get_finished_spans(), "llm.complete").attributes or {}
    assert attrs["openinference.span.kind"] == "LLM"
    assert (attrs["prompt.id"], attrs["prompt.version"], attrs["prompt.hash"]) == (
        "greet",
        2,
        "abc123",
    )
    assert attrs["llm.model_name"] == "m"
    assert attrs["llm.token_count.total"] == 0
    assert json.loads(str(attrs["input.value"])) == [{"role": "user", "content": "hello"}]
    assert attrs["output.value"] == "mock response"


class LlmModule:
    name = "llm_mod"

    def router(self, ctx: PlatformContext) -> APIRouter:
        router = APIRouter()

        @router.get("/call")
        async def call() -> dict[str, str]:
            return {"text": (await ctx.llm.complete(REQUEST)).text}

        return router

    def eval_suites(self) -> list[EvalSuite]:
        return []


@pytest.fixture
def llm_client(exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Created at fixture setup: create_app() reconfigures root logging, and caplog's handler
    # is attached only for the test call phase.
    modules: list[Module] = [LlmModule()]
    monkeypatch.setattr(llmp.api.app, "discover_modules", lambda: modules)
    return TestClient(create_app(Settings(_env_file=None, llm_backend="mock")))


def test_request_log_and_llm_span_share_the_server_trace(
    llm_client: TestClient, exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    client = llm_client
    caplog.handler.setFormatter(JsonFormatter())

    client.get("/llm-mod/call")

    spans = exporter.get_finished_spans()
    server, llm = by_name(spans, "GET /llm-mod/call"), by_name(spans, "llm.complete")
    assert server.context is not None
    assert llm.parent is not None
    assert llm.parent.span_id == server.context.span_id
    (log_entry,) = [
        e for e in map(json.loads, caplog.text.splitlines()) if e["message"] == "request"
    ]
    assert log_entry["trace_id"] == format(server.context.trace_id, "032x")


class NotReadyClient(MockLLMClient):
    async def ready(self) -> bool:
        return False


def test_health_is_503_until_backend_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llmp.api.app, "build_llm_client", lambda *_: NotReadyClient("schema", model="m")
    )
    resp = TestClient(create_app(Settings(_env_file=None, llm_backend="mock"))).get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"


def test_health_probe_is_not_traced(llm_client: TestClient, exporter: InMemorySpanExporter) -> None:
    llm_client.get("/health")
    assert exporter.get_finished_spans() == ()

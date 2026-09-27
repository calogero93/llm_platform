import json

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

import llmp.api.app
from llmp.api.app import create_app
from llmp.config import Settings
from llmp.eval import EvalSuite
from llmp.logs import JsonFormatter
from llmp.modules import Module, PlatformContext


class PingModule:
    name = "ping_mod"

    def router(self, ctx: PlatformContext) -> APIRouter:
        router = APIRouter()

        @router.get("/ping")
        async def ping() -> dict[str, str]:
            return {"backend": ctx.settings.llm_backend}

        return router

    def eval_suites(self) -> list[EvalSuite]:
        return []


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    modules: list[Module] = [PingModule()]
    monkeypatch.setattr(llmp.api.app, "discover_modules", lambda: modules)
    return TestClient(create_app(Settings(_env_file=None, llm_backend="mock")))


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "llm_backend": "mock", "modules": ["ping_mod"]}


def test_module_router_is_mounted_under_its_name(client: TestClient) -> None:
    assert client.get("/ping-mod/ping").json() == {"backend": "mock"}


def test_request_id_is_echoed(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("bad", [b"", b"has space", b"x" * 129, "evil\u2028id".encode()])
def test_invalid_request_id_is_replaced(client: TestClient, bad: bytes) -> None:
    resp = client.get("/health", headers={"X-Request-ID": bad})
    assert len(resp.headers["X-Request-ID"]) == 32  # fresh uuid4 hex


def test_request_log_is_json_with_correlation_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.handler.setFormatter(JsonFormatter())  # format at emit time, as in production
    client.get("/health", headers={"X-Request-ID": "trace-me"})
    entries = [json.loads(line) for line in caplog.text.splitlines()]
    (entry,) = [e for e in entries if e["message"] == "request"]
    assert entry["correlation_id"] == "trace-me"
    assert (entry["method"], entry["path"], entry["status"]) == ("GET", "/health", 200)

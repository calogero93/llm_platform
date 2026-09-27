import httpx
from opentelemetry.trace import Tracer

from llmp.config import Settings
from llmp.llm.base import LLMClient
from llmp.llm.mock import CassetteStore, MockLLMClient
from llmp.llm.vllm import VLLMClient
from llmp.tracing import TracedLLMClient


def build_llm_client(settings: Settings, tracer: Tracer) -> LLMClient:
    return TracedLLMClient(_build_backend(settings), tracer)


def _build_backend(settings: Settings) -> LLMClient:
    if settings.llm_backend == "vllm":
        http = httpx.AsyncClient(
            base_url=str(settings.vllm_base_url), timeout=settings.llm_timeout_s
        )
        return VLLMClient(http, model=settings.llm_model)
    cassettes = CassetteStore(settings.cassette_dir) if settings.mock_mode == "replay" else None
    return MockLLMClient(settings.mock_mode, model=settings.llm_model, cassettes=cassettes)

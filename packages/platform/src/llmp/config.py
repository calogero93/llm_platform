from pathlib import Path
from typing import Literal

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform configuration, read from `LLMP_*` environment variables or `.env`."""

    # env_ignore_empty: compose passes unset optional values as "" (e.g. no OTLP endpoint).
    model_config = SettingsConfigDict(
        env_prefix="LLMP_", env_file=".env", extra="ignore", env_ignore_empty=True
    )

    llm_backend: Literal["vllm", "mock"]
    llm_model: str = "default"
    llm_timeout_s: float = 120.0
    vllm_base_url: HttpUrl = HttpUrl("http://vllm:8000")
    mock_mode: Literal["schema", "replay"] = "schema"
    cassette_dir: Path = Path("cassettes")
    log_level: str = "INFO"
    service_name: str = "llmp-api"
    # OTLP/HTTP traces endpoint, e.g. http://phoenix:6006/v1/traces. None = spans not exported.
    otlp_traces_endpoint: HttpUrl | None = None

from pathlib import Path
from typing import Literal

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform configuration, read from `LLMP_*` environment variables or `.env`."""

    model_config = SettingsConfigDict(env_prefix="LLMP_", env_file=".env", extra="ignore")

    llm_backend: Literal["vllm", "mock"]
    llm_model: str = "default"
    llm_timeout_s: float = 120.0
    vllm_base_url: HttpUrl = HttpUrl("http://vllm:8000")
    mock_mode: Literal["schema", "replay"] = "schema"
    cassette_dir: Path = Path("cassettes")
    log_level: str = "INFO"

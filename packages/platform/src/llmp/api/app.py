from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from llmp.api.middleware import CorrelationIdMiddleware
from llmp.config import Settings
from llmp.llm.factory import build_llm_client
from llmp.logs import configure_logging
from llmp.modules import PlatformContext, discover_modules
from llmp.tracing import build_tracer_provider


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    llm_backend: str
    modules: list[str]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()  # values come from env
    configure_logging(settings.log_level)
    tracer_provider = build_tracer_provider(settings)
    tracer = tracer_provider.get_tracer("llmp")
    ctx = PlatformContext(settings=settings, llm=build_llm_client(settings, tracer))
    modules = discover_modules()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await ctx.llm.aclose()
        tracer_provider.shutdown()  # flushes pending spans

    app = FastAPI(title="LLM Platform", lifespan=lifespan)
    app.add_middleware(CorrelationIdMiddleware, tracer=tracer)

    @app.get("/health")
    async def health(response: Response) -> Health:
        """Readiness: 503 until the LLM backend can serve (e.g. vLLM still loading weights)."""
        ready = await ctx.llm.ready()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Health(
            status="ok" if ready else "unavailable",
            llm_backend=settings.llm_backend,
            modules=[m.name for m in modules],
        )

    for module in modules:
        app.include_router(module.router(ctx), prefix=f"/{module.name.replace('_', '-')}")
    return app

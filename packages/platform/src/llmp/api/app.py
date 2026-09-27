from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from llmp.api.middleware import CorrelationIdMiddleware
from llmp.config import Settings
from llmp.llm.factory import build_llm_client
from llmp.logs import configure_logging
from llmp.modules import PlatformContext, discover_modules


class Health(BaseModel):
    status: str
    llm_backend: str
    modules: list[str]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()  # values come from env
    configure_logging(settings.log_level)
    ctx = PlatformContext(settings=settings, llm=build_llm_client(settings))
    modules = discover_modules()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await ctx.llm.aclose()

    app = FastAPI(title="LLM Platform", lifespan=lifespan)
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/health")
    async def health() -> Health:
        return Health(
            status="ok", llm_backend=settings.llm_backend, modules=[m.name for m in modules]
        )

    for module in modules:
        app.include_router(module.router(ctx), prefix=f"/{module.name.replace('_', '-')}")
    return app

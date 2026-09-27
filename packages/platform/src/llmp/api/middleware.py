import logging
import re
import time
import uuid

from opentelemetry import trace
from opentelemetry.trace import INVALID_SPAN, SpanKind, StatusCode, Tracer
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llmp.logs import correlation_id

HEADER = "X-Request-ID"
# Client-supplied ids end up in logs: accept only a safe, bounded charset.
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Polled every few seconds by orchestrators: a trace per probe would drown the useful ones.
_UNTRACED_PATHS = {"/health"}

log = logging.getLogger(__name__)


class CorrelationIdMiddleware:
    """Per request: sets and echoes the correlation id, opens the root server span (so every log
    line and LLM span of the request share one trace id), and logs one summary line."""

    def __init__(self, app: ASGIApp, tracer: Tracer) -> None:
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(HEADER.lower().encode(), b"").decode("latin-1")
        cid = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        token = correlation_id.set(cid)
        start = time.perf_counter()
        status = 500

        async def send_with_header(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append(HEADER, cid)
            await send(message)

        span_name = f"{scope['method']} {scope['path']}"
        with (
            trace.use_span(INVALID_SPAN)
            if scope["path"] in _UNTRACED_PATHS
            else self.tracer.start_as_current_span(span_name, kind=SpanKind.SERVER)
        ) as span:
            span.set_attribute("correlation_id", cid)
            try:
                await self.app(scope, receive, send_with_header)
            finally:
                span.set_attribute("http.response.status_code", status)
                if status >= 500:
                    span.set_status(StatusCode.ERROR)
                self._log(scope, status, start)
                correlation_id.reset(token)

    @staticmethod
    def _log(scope: Scope, status: int, start: float) -> None:
        log.info(
            "request",
            extra={
                "method": scope["method"],
                "path": scope["path"],
                "status": status,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )

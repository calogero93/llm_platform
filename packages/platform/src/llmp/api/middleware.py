import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llmp.logs import correlation_id

HEADER = "X-Request-ID"
# Client-supplied ids end up in logs: accept only a safe, bounded charset.
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

log = logging.getLogger(__name__)


class CorrelationIdMiddleware:
    """Sets the correlation id for the request, echoes it back, and logs one line per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

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

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            correlation_id.reset(token)

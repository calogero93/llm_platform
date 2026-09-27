"""End-to-end smoke test of the configured LLM backend: `python -m llmp.smoke`.

Sends one schema-constrained request through the traced client (the same path modules use) and
prints the parsed answer and trace id. Exits non-zero if the output does not match the schema.
"""

import asyncio
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from llmp.config import Settings
from llmp.llm import LLMRequest, Message
from llmp.llm.factory import build_llm_client
from llmp.logs import configure_logging
from llmp.prompts import PromptStore
from llmp.tracing import build_tracer_provider

TEXT = "Fattura n. 2026/118 emessa dalla sede di Bologna, via Emilia 12, il 3 marzo 2026."

log = logging.getLogger(__name__)


class IssuingCity(BaseModel):
    city: str
    province_code: str = Field(min_length=2, max_length=2)


async def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    provider = build_tracer_provider(settings)
    tracer = provider.get_tracer("llmp.smoke")
    client = build_llm_client(settings, tracer)
    prompt = PromptStore(Path(__file__).parent / "templates").get("smoke", 1)
    try:
        with tracer.start_as_current_span("smoke") as span:
            response = await client.complete(
                LLMRequest(
                    messages=(
                        Message(role="system", content=prompt.render()),
                        Message(role="user", content=TEXT),
                    ),
                    prompt=prompt.ref,
                    output_model=IssuingCity,
                    max_tokens=64,
                )
            )
            answer = response.parse(IssuingCity)
            trace_id = format(span.get_span_context().trace_id, "032x")
            log.info(
                "smoke completed",
                extra={
                    "answer": answer.model_dump(),
                    "model": response.model,
                    "latency_s": round(response.latency_s, 3),
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        print(f"answer={answer.model_dump()} trace_id={trace_id}")
    finally:
        await client.aclose()
        provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

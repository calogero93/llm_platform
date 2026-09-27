"""OpenTelemetry setup and the tracing wrapper around `LLMClient`.

Spans use OpenInference attribute names (`openinference.span.kind`, `llm.*`, `input.value`, ...)
so Phoenix renders them as LLM calls, but only the plain OTel SDK is used: any OTLP backend
(e.g. Langfuse) can receive them by changing the endpoint.
"""

import json

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from llmp.config import Settings
from llmp.llm.base import LLMClient
from llmp.llm.types import LLMRequest, LLMResponse


def build_tracer_provider(settings: Settings) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    if settings.otlp_traces_endpoint is not None:
        exporter = OTLPSpanExporter(endpoint=str(settings.otlp_traces_endpoint))
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


class TracedLLMClient:
    """Records one LLM span per completion: prompt version, model, tokens, input and output."""

    def __init__(self, inner: LLMClient, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    async def complete(self, request: LLMRequest) -> LLMResponse:
        with self._tracer.start_as_current_span("llm.complete") as span:
            prompt = request.prompt
            span.set_attributes(
                {
                    "openinference.span.kind": "LLM",
                    "prompt.id": prompt.id,
                    "prompt.version": prompt.version,
                    "prompt.hash": prompt.hash,
                    "llm.prompt_template.version": f"{prompt.id}:v{prompt.version}:{prompt.hash}",
                    "llm.invocation_parameters": json.dumps(
                        {"temperature": request.temperature, "max_tokens": request.max_tokens}
                    ),
                    "input.mime_type": "application/json",
                    "input.value": json.dumps(
                        [m.model_dump() for m in request.messages], ensure_ascii=False
                    ),
                }
            )
            for i, message in enumerate(request.messages):
                span.set_attribute(f"llm.input_messages.{i}.message.role", message.role)
                span.set_attribute(f"llm.input_messages.{i}.message.content", message.content)
            if request.output_model is not None:
                span.set_attribute("llm.output_schema", request.output_model.__name__)

            response = await self._inner.complete(request)

            usage = response.usage
            span.set_attributes(
                {
                    "llm.model_name": response.model,
                    "llm.token_count.prompt": usage.prompt_tokens,
                    "llm.token_count.completion": usage.completion_tokens,
                    "llm.token_count.total": usage.prompt_tokens + usage.completion_tokens,
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": response.text,
                    "output.value": response.text,
                }
            )
            return response

    async def ready(self) -> bool:
        return await self._inner.ready()

    async def aclose(self) -> None:
        await self._inner.aclose()

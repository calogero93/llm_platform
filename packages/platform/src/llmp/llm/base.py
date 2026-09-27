from typing import Protocol, runtime_checkable

from llmp.llm.types import LLMRequest, LLMResponse


class LLMError(Exception):
    """The LLM backend failed to produce a response."""


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def ready(self) -> bool:
        """Whether the backend can serve requests now (model loaded)."""
        ...

    async def aclose(self) -> None: ...

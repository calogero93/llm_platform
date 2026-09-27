from typing import Protocol

from llmp.llm.types import LLMRequest, LLMResponse


class LLMError(Exception):
    """The LLM backend failed to produce a response."""


class LLMClient(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def aclose(self) -> None: ...

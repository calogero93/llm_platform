from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class PromptRef:
    """Identifies the prompt template a request was built from (recorded in traces and evals)."""

    id: str
    version: int
    hash: str


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[Message, ...]
    prompt: PromptRef
    output_model: type[BaseModel] | None = None
    """When set, generation is constrained to this model's JSON schema."""
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage
    latency_s: float

    def parse[T: BaseModel](self, output_model: type[T]) -> T:
        """Validate the response text against a pydantic model."""
        return output_model.model_validate_json(self.text)

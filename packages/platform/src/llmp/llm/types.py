from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class PromptRef(_Frozen):
    """Identifies the prompt template a request was built from (recorded in traces and evals)."""

    id: str
    version: int
    hash: str


class Message(_Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(_Frozen):
    messages: tuple[Message, ...]
    prompt: PromptRef
    # When set, generation is constrained to this model's JSON schema.
    output_model: type[BaseModel] | None = None
    temperature: float = 0.0
    max_tokens: int = 1024


class Usage(_Frozen):
    prompt_tokens: int
    completion_tokens: int


class LLMResponse(_Frozen):
    text: str
    model: str
    usage: Usage
    latency_s: float

    def parse[T: BaseModel](self, output_model: type[T]) -> T:
        """Validate the response text against a pydantic model."""
        return output_model.model_validate_json(self.text)

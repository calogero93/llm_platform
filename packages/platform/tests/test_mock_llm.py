from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from llmp.llm import LLMRequest, LLMResponse, Message, PromptRef, Usage
from llmp.llm.mock import CassetteMissError, CassetteStore, MockLLMClient, cassette_key

PROMPT = PromptRef(id="test", version=1, hash="abc")


class Kind(StrEnum):
    INVOICE = "invoice"
    DDT = "ddt"


class Line(BaseModel):
    description: str = Field(min_length=3)
    quantity: int = Field(ge=1)
    unit_price: Decimal


class Doc(BaseModel):
    kind: Kind
    number: str
    issued_on: date
    currency: Literal["EUR"]
    notes: str | None
    lines: list[Line] = Field(min_length=2)
    total: Decimal = Field(gt=0)


def request(content: str = "extract", output_model: type[BaseModel] | None = Doc) -> LLMRequest:
    return LLMRequest(
        messages=(Message(role="user", content=content),), prompt=PROMPT, output_model=output_model
    )


async def test_schema_mode_returns_valid_instance() -> None:
    client = MockLLMClient("schema", model="m")
    doc = (await client.complete(request())).parse(Doc)
    assert len(doc.lines) == 2
    assert doc.total > 0


async def test_schema_mode_is_deterministic() -> None:
    client = MockLLMClient("schema", model="m")
    assert (await client.complete(request())).text == (await client.complete(request())).text


async def test_schema_mode_without_output_model_returns_text() -> None:
    client = MockLLMClient("schema", model="m")
    assert (await client.complete(request(output_model=None))).text == "mock response"


async def test_schema_mode_rejects_unsupported_pattern() -> None:
    class Coded(BaseModel):
        code: str = Field(pattern=r"^\d{11}$")

    with pytest.raises(ValueError, match="pattern"):
        await MockLLMClient("schema", model="m").complete(request(output_model=Coded))


async def test_replay_returns_recorded_response(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    recorded = LLMResponse(
        text='{"a": 1}',
        model="m",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        latency_s=0.42,
    )
    store.save(cassette_key(request(), "m"), request(), recorded)

    client = MockLLMClient("replay", model="m", cassettes=store)
    assert await client.complete(request()) == recorded
    assert await client.complete(request()) == recorded


async def test_replay_miss_raises(tmp_path: Path) -> None:
    client = MockLLMClient("replay", model="m", cassettes=CassetteStore(tmp_path))
    with pytest.raises(CassetteMissError, match="re-record"):
        await client.complete(request())


def test_cassette_key_changes_with_any_generation_input() -> None:
    base = cassette_key(request(), "m")
    assert cassette_key(request(), "m") == base
    assert cassette_key(request(content="other"), "m") != base
    assert cassette_key(request(output_model=Line), "m") != base
    assert cassette_key(request(), "other-model") != base

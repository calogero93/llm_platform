"""Versioned prompt templates stored as files: `<root>/<prompt_id>/v<N>.md`.

Templates use `string.Template` placeholders (`$name`). A version file must never be edited once
it has been used in a committed eval baseline: behaviour changes go in a new version file.
"""

import hashlib
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict

from llmp.llm.types import PromptRef


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref: PromptRef
    text: str

    def render(self, **values: str) -> str:
        """Substitute all placeholders; a missing value raises `KeyError`."""
        return Template(self.text).substitute(values)


class PromptNotFoundError(LookupError):
    pass


class PromptStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def get(self, prompt_id: str, version: int) -> Prompt:
        path = self._root / prompt_id / f"v{version}.md"
        if not path.is_file():
            raise PromptNotFoundError(f"prompt {prompt_id!r} v{version} not found at {path}")
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        return Prompt(ref=PromptRef(id=prompt_id, version=version, hash=digest), text=text)

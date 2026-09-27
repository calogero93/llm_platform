from pathlib import Path

import pytest

from llmp.prompts import PromptNotFoundError, PromptStore


@pytest.fixture
def store(tmp_path: Path) -> PromptStore:
    (tmp_path / "greet").mkdir()
    (tmp_path / "greet" / "v1.md").write_text("Hello $name", encoding="utf-8")
    (tmp_path / "greet" / "v2.md").write_text("Hi $name!", encoding="utf-8")
    return PromptStore(tmp_path)


def test_render_substitutes_placeholders(store: PromptStore) -> None:
    assert store.get("greet", 1).render(name="Ada") == "Hello Ada"


def test_missing_placeholder_value_raises(store: PromptStore) -> None:
    with pytest.raises(KeyError):
        store.get("greet", 1).render()


def test_ref_identifies_id_version_and_content(store: PromptStore) -> None:
    v1, v2 = store.get("greet", 1).ref, store.get("greet", 2).ref
    assert (v1.id, v1.version) == ("greet", 1)
    assert v1.hash == store.get("greet", 1).ref.hash
    assert v1.hash != v2.hash


def test_unknown_version_raises(store: PromptStore) -> None:
    with pytest.raises(PromptNotFoundError):
        store.get("greet", 3)

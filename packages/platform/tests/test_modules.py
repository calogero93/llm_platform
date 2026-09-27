from dataclasses import dataclass

import pytest
from fastapi import APIRouter

import llmp.modules
from llmp.modules import ModuleLoadError, PlatformContext, discover_modules


class FakeModule:
    def __init__(self, name: str) -> None:
        self.name = name

    def router(self, ctx: PlatformContext) -> APIRouter:
        return APIRouter()


@dataclass
class FakeEntryPoint:
    value: str
    obj: object

    def load(self) -> object:
        return self.obj


def patch_entry_points(monkeypatch: pytest.MonkeyPatch, *objs: object) -> None:
    eps = [FakeEntryPoint(f"pkg:obj{i}", o) for i, o in enumerate(objs)]
    monkeypatch.setattr(llmp.modules, "entry_points", lambda group: eps)


def test_discovers_modules_sorted_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, FakeModule("b"), FakeModule("a"))
    assert [m.name for m in discover_modules()] == ["a", "b"]


def test_rejects_object_not_implementing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, object())
    with pytest.raises(ModuleLoadError, match="does not implement"):
        discover_modules()


def test_rejects_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, FakeModule("a"), FakeModule("a"))
    with pytest.raises(ModuleLoadError, match="duplicate"):
        discover_modules()

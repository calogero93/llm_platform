"""Module contract and discovery.

A module is an installed distribution exposing an object under the `llmp.modules` entry-point
group. The platform discovers modules at startup; it never imports them by name.
"""

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from fastapi import APIRouter

from llmp.config import Settings
from llmp.llm.base import LLMClient

ENTRY_POINT_GROUP = "llmp.modules"


@dataclass(frozen=True)
class PlatformContext:
    """Services the platform hands to modules. Modules must not build these themselves."""

    settings: Settings
    llm: LLMClient


@runtime_checkable
class Module(Protocol):
    name: str

    def router(self, ctx: PlatformContext) -> APIRouter: ...


class ModuleLoadError(Exception):
    pass


def discover_modules() -> list[Module]:
    modules: dict[str, Module] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        obj = ep.load()
        if not isinstance(obj, Module):
            raise ModuleLoadError(f"entry point {ep.value!r} does not implement Module")
        if obj.name in modules:
            raise ModuleLoadError(f"duplicate module name {obj.name!r}")
        modules[obj.name] = obj
    return [modules[name] for name in sorted(modules)]

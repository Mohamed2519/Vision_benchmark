"""
Model registry — the single place where models are registered.

Usage
-----
from benchmarks.registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseVisionModel):
    ...
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type

from models.base import BaseVisionModel


class ModelRegistry:
    _registry: Dict[str, Type[BaseVisionModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a model class under *name*."""
        def decorator(model_cls: Type[BaseVisionModel]):
            if name in cls._registry:
                raise ValueError(f"Model '{name}' is already registered.")
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseVisionModel]:
        if name not in cls._registry:
            raise KeyError(
                f"Model '{name}' not found. Available: {list(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def autodiscover(cls, package: str = "models") -> None:
        """Import every module inside *package* so decorators fire."""
        pkg_path = Path(__file__).parent.parent / package
        for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            if not module_name.startswith("_"):
                importlib.import_module(f"{package}.{module_name}")

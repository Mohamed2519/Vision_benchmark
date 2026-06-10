import pytest
from benchmarks.registry import ModelRegistry
from models.base import BaseVisionModel


def test_register_and_get():
    @ModelRegistry.register("_test_model_xyz")
    class DummyModel(BaseVisionModel):
        def predict(self, images):
            return [{"label": "cat", "conf": 1.0} for _ in images]

    cls = ModelRegistry.get("_test_model_xyz")
    assert cls is DummyModel


def test_duplicate_raises():
    @ModelRegistry.register("_test_dup")
    class M1(BaseVisionModel):
        def predict(self, images): return []

    with pytest.raises(ValueError):
        @ModelRegistry.register("_test_dup")
        class M2(BaseVisionModel):
            def predict(self, images): return []


def test_unknown_raises():
    with pytest.raises(KeyError):
        ModelRegistry.get("does_not_exist_xyz")

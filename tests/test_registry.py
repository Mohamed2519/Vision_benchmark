import pytest
from benchmarks.registry import ModelRegistry
from models.base import ChestXrayModel


def test_register_and_get():
    @ModelRegistry.register("_test_model_xyz")
    class DummyModel(ChestXrayModel):
        def predict_batch(self, image_paths):
            return [{"score": 1.0, "label": 1} for _ in image_paths]

    cls = ModelRegistry.get("_test_model_xyz")
    assert cls is DummyModel


def test_duplicate_raises():
    @ModelRegistry.register("_test_dup")
    class M1(ChestXrayModel):
        def predict_batch(self, image_paths): return []

    with pytest.raises(ValueError):
        @ModelRegistry.register("_test_dup")
        class M2(ChestXrayModel):
            def predict_batch(self, image_paths): return []


def test_unknown_raises():
    with pytest.raises(KeyError):
        ModelRegistry.get("does_not_exist_xyz")

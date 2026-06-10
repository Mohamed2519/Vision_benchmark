import sys
import types

import pytest
from click.testing import CliRunner

import benchmark


def test_run_defaults_to_all_models_all_datasets_cuda(monkeypatch):
    run_calls = []

    class DummyEvaluator:
        def __init__(self, config_path, datasets_config):
            pass

        def run_single(self, **kwargs):
            run_calls.append(kwargs)

        @property
        def loader(self):
            class _Loader:
                def dataset_names(self):
                    return ["dataset1", "dataset2"]
            return _Loader()

    class DummyRegistry:
        @classmethod
        def autodiscover(cls, package="models"):
            return None

        @classmethod
        def list_models(cls):
            return ["model_a", "model_b"]

    monkeypatch.setattr("benchmark.ModelRegistry", DummyRegistry, raising=False)
    # Patch inside the function's import scope
    monkeypatch.setitem(sys.modules, "benchmarks.registry",
                        types.SimpleNamespace(ModelRegistry=DummyRegistry))
    monkeypatch.setitem(sys.modules, "evaluate",
                        types.SimpleNamespace(Evaluator=DummyEvaluator))

    result = CliRunner().invoke(
        benchmark.cli,
        ["--datasets-config", "custom_datasets.yaml", "run"],
    )

    assert result.exit_code == 0, result.output
    assert len(run_calls) == 4
    assert {(c["model_name"], c["dataset_name"]) for c in run_calls} == {
        ("model_a", "dataset1"),
        ("model_a", "dataset2"),
        ("model_b", "dataset1"),
        ("model_b", "dataset2"),
    }
    assert all(c["device"] == "cuda" for c in run_calls)


def test_run_with_explicit_models_and_dataset(monkeypatch):
    run_calls = []

    class DummyEvaluator:
        def __init__(self, config_path, datasets_config):
            pass

        def run_single(self, **kwargs):
            run_calls.append(kwargs)

        @property
        def loader(self):
            class _Loader:
                def dataset_names(self):
                    return ["dataset1"]
            return _Loader()

    monkeypatch.setitem(sys.modules, "evaluate",
                        types.SimpleNamespace(Evaluator=DummyEvaluator))

    result = CliRunner().invoke(
        benchmark.cli,
        ["run", "--models", "model_x", "--models", "model_y",
         "--datasets", "cifar10", "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    assert len(run_calls) == 2
    assert {(c["model_name"], c["dataset_name"]) for c in run_calls} == {
        ("model_x", "cifar10"),
        ("model_y", "cifar10"),
    }
    assert all(c["device"] == "cpu" for c in run_calls)

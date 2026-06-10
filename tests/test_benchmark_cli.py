import sys
import types

from click.testing import CliRunner

import benchmark


def test_run_defaults_to_all_models_all_datasets_cuda(monkeypatch):
    run_calls = []
    runner_inits = []
    loader_paths = []

    class DummyRunner:
        def __init__(self, config_path, datasets_config_path):
            runner_inits.append((config_path, datasets_config_path))

        def run(self, **kwargs):
            run_calls.append(kwargs)

    class DummyRegistry:
        @classmethod
        def autodiscover(cls):
            return None

        @classmethod
        def list_models(cls):
            return ["model_a", "model_b"]

    class DummyLoader:
        def __init__(self, config_path):
            loader_paths.append(config_path)

        def dataset_names(self):
            return ["dataset1", "dataset2"]

    monkeypatch.setitem(sys.modules, "benchmarks.runner", types.SimpleNamespace(BenchmarkRunner=DummyRunner))
    monkeypatch.setitem(sys.modules, "benchmarks.registry", types.SimpleNamespace(ModelRegistry=DummyRegistry))
    monkeypatch.setitem(sys.modules, "datasets.loader", types.SimpleNamespace(DatasetLoader=DummyLoader))

    result = CliRunner().invoke(benchmark.cli, ["--datasets-config", "custom_datasets.yaml", "run"])

    assert result.exit_code == 0
    assert runner_inits == [("configs/default.yaml", "custom_datasets.yaml")]
    assert loader_paths == ["custom_datasets.yaml"]
    assert len(run_calls) == 4
    assert {(c["model_name"], c["dataset"]) for c in run_calls} == {
        ("model_a", "dataset1"),
        ("model_a", "dataset2"),
        ("model_b", "dataset1"),
        ("model_b", "dataset2"),
    }
    assert all(c["model_kwargs"] == {"device": "cuda"} for c in run_calls)


def test_run_with_explicit_models_and_dataset(monkeypatch):
    run_calls = []

    class DummyRunner:
        def __init__(self, config_path, datasets_config_path):
            pass

        def run(self, **kwargs):
            run_calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "benchmarks.runner", types.SimpleNamespace(BenchmarkRunner=DummyRunner))

    result = CliRunner().invoke(
        benchmark.cli,
        ["run", "model_x", "model_y", "--dataset", "cifar10", "--device", "cpu"],
    )

    assert result.exit_code == 0
    assert len(run_calls) == 2
    assert {(c["model_name"], c["dataset"]) for c in run_calls} == {
        ("model_x", "cifar10"),
        ("model_y", "cifar10"),
    }
    assert all(c["model_kwargs"] == {"device": "cpu"} for c in run_calls)

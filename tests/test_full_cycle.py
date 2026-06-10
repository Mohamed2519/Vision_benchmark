"""
Full-cycle integration test for the chest X-ray benchmark pipeline.

Covers the complete flow using synthetic data and a dummy model:
  1. DatasetLoader  — loads val+test and golden splits
  2. BinaryClassificationTask — runs inference and collects predictions
  3. compute_metrics / combine_scores — all metrics computed correctly
  4. ResultStore — persist + retrieve runs and predictions
  5. ErrorAnalyzer — per-class breakdown, top failures, confusion pairs, ECE
  6. HTML report generation — smoke test (no crash)
  7. evaluate.py Evaluator — end-to-end orchestration with a registered model

All disk I/O is redirected to a tmp_path fixture so the test is self-contained.
"""
from __future__ import annotations

import csv
import json
import os
import textwrap
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path: Path) -> None:
    """Create a minimal valid JPEG."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(str(path))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_dataset(tmp_path):
    """
    Build a minimal on-disk dataset:
      - 20 val+test images in a CSV
      - 10 golden images with .txt annotation files
    Returns a dict with all relevant paths.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    csv_rows = []
    for i in range(20):
        p = images_dir / f"sample_{i:03d}.jpg"
        _make_image(p)
        label = i % 2          # alternating 0/1
        csv_rows.append({"Path": str(p), "label": label})

    csv_path = tmp_path / "val_test.csv"
    _write_csv(csv_path, csv_rows)

    # Golden set
    gold_images = tmp_path / "golden_images"
    gold_images.mkdir()
    gold_labels = tmp_path / "golden_labels"
    gold_labels.mkdir()

    for i in range(10):
        p = gold_images / f"gold_{i:03d}.jpg"
        _make_image(p)
        label = i % 2
        (gold_labels / f"gold_{i:03d}.txt").write_text(str(label))

    return {
        "csv_path": csv_path,
        "images_dir": str(gold_images),
        "annotations_dir": str(gold_labels),
        "tmp": tmp_path,
    }


@pytest.fixture()
def datasets_config(synthetic_dataset, tmp_path):
    """Write a minimal datasets.yaml pointing at the synthetic data."""
    cfg = {
        "datasets": {
            "test_ds": {
                "val_test": {
                    "csv": [str(synthetic_dataset["csv_path"])],
                },
                "golden": {
                    "images_dir": synthetic_dataset["images_dir"],
                    "annotations_dir": synthetic_dataset["annotations_dir"],
                },
            },
            "no_golden_ds": {
                "val_test": {
                    "csv": [str(synthetic_dataset["csv_path"])],
                },
                "golden": {
                    "images_dir": None,
                    "annotations_dir": None,
                },
            },
        }
    }
    p = tmp_path / "datasets.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


@pytest.fixture()
def default_config(tmp_path):
    """Write a minimal default.yaml."""
    cfg = {
        "benchmark": {"name": "test", "output_dir": str(tmp_path / "results")},
        "wandb": {"enabled": False},
        "error_analysis": {
            "enabled": True,
            "top_n_failures": 100,
            "confidence_bins": 10,
            "per_class_breakdown": True,
        },
    }
    p = tmp_path / "default.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


# ---------------------------------------------------------------------------
# Unit: DatasetLoader
# ---------------------------------------------------------------------------

class TestDatasetLoader:
    def test_val_test_loads_correct_count(self, datasets_config):
        from datasets.loader import DatasetLoader
        from datasets.base import Split

        loader = DatasetLoader(datasets_config)
        records = loader.load("test_ds", Split.VAL_TEST)
        assert len(records) == 20

    def test_golden_loads_correct_count(self, datasets_config):
        from datasets.loader import DatasetLoader
        from datasets.base import Split

        loader = DatasetLoader(datasets_config)
        records = loader.load("test_ds", Split.GOLDEN)
        assert len(records) == 10

    def test_missing_golden_returns_empty(self, datasets_config):
        import warnings
        from datasets.loader import DatasetLoader
        from datasets.base import Split

        loader = DatasetLoader(datasets_config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            records = loader.load("no_golden_ds", Split.GOLDEN)

        assert records == []
        assert any("No golden set" in str(x.message) for x in w)

    def test_dataset_names(self, datasets_config):
        from datasets.loader import DatasetLoader

        loader = DatasetLoader(datasets_config)
        names = loader.dataset_names()
        assert set(names) == {"test_ds", "no_golden_ds"}


# ---------------------------------------------------------------------------
# Unit: compute_metrics + combine_scores
# ---------------------------------------------------------------------------

class TestBinaryMetrics:
    def test_perfect_predictions(self):
        from tasks.binary_metrics import compute_metrics

        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 0, 1, 1, 0, 1]
        y_score = [0.1, 0.2, 0.8, 0.9, 0.1, 0.7]
        m = compute_metrics(y_true, y_pred, y_score)

        assert m["accuracy"] == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["auroc"] == pytest.approx(1.0)

    def test_all_wrong(self):
        from tasks.binary_metrics import compute_metrics

        y_true  = [0, 0, 1, 1]
        y_pred  = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.1, 0.2]
        m = compute_metrics(y_true, y_pred, y_score)

        assert m["accuracy"] == pytest.approx(0.0)
        assert m["tp"] == 0
        assert m["tn"] == 0

    def test_combine_scores_all_vt(self):
        from tasks.binary_metrics import combine_scores

        pairs = [(0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9)]
        auroc = combine_scores(pairs, [])
        assert auroc == pytest.approx(1.0)

    def test_combine_scores_mixed(self):
        from tasks.binary_metrics import combine_scores

        vt = [(0, 0.1), (1, 0.9)]
        g  = [(0, 0.2), (1, 0.8)]
        auroc = combine_scores(vt, g)
        assert 0.0 <= auroc <= 1.0


# ---------------------------------------------------------------------------
# Unit: BinaryClassificationTask
# ---------------------------------------------------------------------------

class DummyModel:
    """Returns alternating 0/1 predictions with confident scores."""
    def __init__(self, device="cpu"):
        self._i = 0

    def predict_batch(self, image_paths):
        results = []
        for _ in image_paths:
            label = self._i % 2
            results.append({"score": 0.9 if label else 0.1, "label": label})
            self._i += 1
        return results


class TestBinaryClassificationTask:
    def test_run_returns_expected_keys(self, datasets_config):
        from datasets.loader import DatasetLoader
        from datasets.base import Split
        from tasks.binary_classification import BinaryClassificationTask

        loader = DatasetLoader(datasets_config)
        records = loader.load("test_ds", Split.VAL_TEST)
        task = BinaryClassificationTask(batch_size=8, threshold=0.5)
        result = task.run(DummyModel(), records, run_id="test_run", desc="test")

        assert "metrics" in result
        assert "predictions" in result
        assert len(result["predictions"]) == len(records)

    def test_predictions_have_required_fields(self, datasets_config):
        from datasets.loader import DatasetLoader
        from datasets.base import Split
        from tasks.binary_classification import BinaryClassificationTask

        loader = DatasetLoader(datasets_config)
        records = loader.load("test_ds", Split.VAL_TEST)
        task = BinaryClassificationTask(batch_size=8, threshold=0.5)
        result = task.run(DummyModel(), records, run_id="pred_fields", desc="t")

        for pred in result["predictions"]:
            assert "label" in pred
            assert "pred" in pred
            assert "confidence" in pred
            assert "correct" in pred


# ---------------------------------------------------------------------------
# Unit: ResultStore
# ---------------------------------------------------------------------------

class TestResultStore:
    def test_save_and_retrieve_run(self, tmp_path):
        from benchmarks.result_store import ResultStore

        store = ResultStore(str(tmp_path / "results.db"))
        metrics = {"auroc": 0.85, "accuracy": 0.80}
        run_id = "abc123"
        store.save_run(
            run_id=run_id,
            model_name="test_model",
            task="binary_classification",
            dataset="test_ds",
            split="combined",
            metrics=metrics,
        )

        df = store.get_runs()
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == run_id
        assert df.iloc[0]["model_name"] == "test_model"

    def test_save_and_retrieve_predictions(self, tmp_path):
        from benchmarks.result_store import ResultStore

        store = ResultStore(str(tmp_path / "results.db"))
        run_id = "pred_run"
        preds = [
            {"run_id": run_id, "sample_id": "s1", "label": 0, "pred": 1, "confidence": 0.8, "correct": 0},
            {"run_id": run_id, "sample_id": "s2", "label": 1, "pred": 1, "confidence": 0.9, "correct": 1},
        ]
        store.save_predictions(preds)
        df = store.get_predictions(run_id)
        assert len(df) == 2

    def test_export_csv(self, tmp_path):
        from benchmarks.result_store import ResultStore

        store = ResultStore(str(tmp_path / "results.db"))
        store.save_run("r1", "m", "t", "d", "combined", {"auroc": 0.7})
        out = str(tmp_path / "runs_summary.csv")
        store.export_csv(path=out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# Unit: ErrorAnalyzer
# ---------------------------------------------------------------------------

class TestErrorAnalyzer:
    def _make_preds(self, n=30):
        rng = np.random.default_rng(42)
        preds = []
        for i in range(n):
            label = i % 2
            wrong = rng.random() < 0.3
            pred  = 1 - label if wrong else label
            conf  = float(rng.uniform(0.6, 0.99))
            preds.append({
                "run_id":     "test_run",
                "sample_id":  str(i),
                "image_path": f"/fake/img_{i}.jpg",
                "label":      label,
                "pred":       pred,
                "confidence": conf,
                "correct":    int(not wrong),
            })
        return preds

    def test_analyze_produces_report(self, tmp_path):
        from error_analysis.analyzer import ErrorAnalyzer

        analyzer = ErrorAnalyzer(
            predictions=self._make_preds(),
            cfg={"top_n_failures": 100, "confidence_bins": 10},
            run_id="test_run",
            output_dir=str(tmp_path),
        )
        report = analyzer.analyze()

        assert "summary" in report
        assert "per_class" in report
        assert "top_failures" in report
        assert "confusion_pairs" in report
        assert "ece" in report

    def test_top_failures_limited_to_100(self, tmp_path):
        from error_analysis.analyzer import ErrorAnalyzer

        preds = self._make_preds(n=60)
        analyzer = ErrorAnalyzer(
            predictions=preds,
            cfg={"top_n_failures": 100},
            run_id="test_run",
            output_dir=str(tmp_path),
        )
        report = analyzer.analyze()
        assert len(report["top_failures"]) <= 100

    def test_confusion_pairs_are_strings_in_rich_table(self, tmp_path):
        """Regression test: Rich table.add_row() requires strings, not ints."""
        from error_analysis.analyzer import ErrorAnalyzer

        preds = self._make_preds(n=20)
        analyzer = ErrorAnalyzer(
            predictions=preds,
            cfg={"top_n_failures": 100},
            run_id="reg_run",
            output_dir=str(tmp_path),
        )
        # _print_report should not raise NotRenderableError
        report = analyzer.analyze()
        assert isinstance(report, dict)

    def test_csvs_written(self, tmp_path):
        from error_analysis.analyzer import ErrorAnalyzer

        analyzer = ErrorAnalyzer(
            predictions=self._make_preds(),
            cfg={"top_n_failures": 100},
            run_id="csv_run",
            output_dir=str(tmp_path),
        )
        analyzer.analyze()

        out = tmp_path / "error_analysis" / "csv_run"
        assert (out / "per_class.csv").exists()
        assert (out / "top_failures.csv").exists()
        assert (out / "confusion_pairs.csv").exists()


# ---------------------------------------------------------------------------
# Integration: Evaluator end-to-end
# ---------------------------------------------------------------------------

class TestEvaluatorEndToEnd:
    def test_run_single_produces_combined_auroc(
        self, datasets_config, default_config, monkeypatch
    ):
        from benchmarks.registry import ModelRegistry
        from evaluate import Evaluator

        # Register dummy model for the test
        ModelRegistry._registry.pop("DUMMY_MODEL", None)
        ModelRegistry.register("DUMMY_MODEL")(DummyModel)

        evaluator = Evaluator(
            config_path=default_config,
            datasets_config=datasets_config,
        )
        result = evaluator.run_single(
            model_name="DUMMY_MODEL",
            dataset_name="test_ds",
            device="cpu",
            batch_size=8,
            threshold=0.5,
        )

        assert "combined_auroc" in result
        assert 0.0 <= result["combined_auroc"] <= 1.0
        assert "val_test_metrics" in result
        assert "golden_metrics" in result

    def test_run_single_no_golden_does_not_crash(
        self, datasets_config, default_config
    ):
        from benchmarks.registry import ModelRegistry
        from evaluate import Evaluator

        ModelRegistry._registry.pop("DUMMY_MODEL", None)
        ModelRegistry.register("DUMMY_MODEL")(DummyModel)

        evaluator = Evaluator(
            config_path=default_config,
            datasets_config=datasets_config,
        )
        result = evaluator.run_single(
            model_name="DUMMY_MODEL",
            dataset_name="no_golden_ds",
            device="cpu",
            batch_size=8,
            threshold=0.5,
        )
        assert result["combined_auroc"] is not None

    def test_results_accumulated_in_db(
        self, datasets_config, default_config, tmp_path
    ):
        from benchmarks.registry import ModelRegistry
        from benchmarks.result_store import ResultStore
        from evaluate import Evaluator
        import yaml

        ModelRegistry._registry.pop("DUMMY_MODEL", None)
        ModelRegistry.register("DUMMY_MODEL")(DummyModel)

        evaluator = Evaluator(
            config_path=default_config,
            datasets_config=datasets_config,
        )
        evaluator.run_single("DUMMY_MODEL", "test_ds", device="cpu", batch_size=8)
        evaluator.run_single("DUMMY_MODEL", "no_golden_ds", device="cpu", batch_size=8)

        cfg = yaml.safe_load(open(default_config))
        store = ResultStore(f"{cfg['benchmark']['output_dir']}/results.db")
        df = store.get_runs()
        # at least 2 combined rows (one per dataset)
        combined = df[df["split"] == "combined"]
        assert len(combined) >= 2


# ---------------------------------------------------------------------------
# Smoke test: HTML report
# ---------------------------------------------------------------------------

class TestHtmlReport:
    def test_report_generates_html(self, datasets_config, default_config, tmp_path):
        import yaml
        from benchmarks.registry import ModelRegistry
        from benchmarks.result_store import ResultStore
        from evaluate import Evaluator
        from click.testing import CliRunner
        from reports.generate_report import generate

        ModelRegistry._registry.pop("DUMMY_MODEL", None)
        ModelRegistry.register("DUMMY_MODEL")(DummyModel)

        evaluator = Evaluator(
            config_path=default_config,
            datasets_config=datasets_config,
        )
        evaluator.run_single("DUMMY_MODEL", "test_ds", device="cpu", batch_size=8)

        cfg = yaml.safe_load(open(default_config))
        db  = f"{cfg['benchmark']['output_dir']}/results.db"
        out = str(tmp_path / "report.html")

        result = CliRunner().invoke(generate, ["--db", db, "--out", out])
        assert result.exit_code == 0, result.output

        assert Path(out).exists()
        content = Path(out).read_text()
        assert "<html" in content.lower()

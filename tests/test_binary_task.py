"""
Tests for the BinaryClassificationTask using a dummy stub model.
"""
import tempfile
from pathlib import Path
from PIL import Image

import pytest

from datasets.base import SampleRecord, Split
from tasks.binary_classification import BinaryClassificationTask


def _make_image(path):
    Image.new("RGB", (32, 32)).save(str(path))


class DummyModel:
    """Always predicts abnormal with score 0.9 for abnormal GT, 0.2 for normal GT."""
    def predict_batch(self, image_paths):
        # We can't know GT from path, so just return alternating
        return [{"score": 0.9, "label": 1} for _ in image_paths]


def make_records(tmp_path, n=6):
    records = []
    for i in range(n):
        p = tmp_path / f"img{i}.jpg"
        _make_image(p)
        records.append(
            SampleRecord(
                image_path=p,
                label=1 if i % 2 == 0 else 0,
                dataset_name="test_ds",
                split=Split.VAL_TEST,
            )
        )
    return records


def test_task_returns_all_predictions(tmp_path):
    records = make_records(tmp_path, n=6)
    task = BinaryClassificationTask(batch_size=2)
    result = task.run(DummyModel(), records, run_id="r1")

    assert len(result["predictions"]) == 6
    assert "metrics" in result


def test_task_metrics_keys(tmp_path):
    records = make_records(tmp_path, n=4)
    task = BinaryClassificationTask()
    result = task.run(DummyModel(), records, run_id="r1")

    metrics = result["metrics"]
    for key in ["accuracy", "sensitivity", "specificity", "f1", "auroc", "auprc"]:
        assert key in metrics, f"Missing metric: {key}"


def test_predictions_have_required_fields(tmp_path):
    records = make_records(tmp_path, n=4)
    task = BinaryClassificationTask()
    result = task.run(DummyModel(), records, run_id="r1")

    for p in result["predictions"]:
        for field in ["run_id", "sample_id", "image_path", "label", "pred", "confidence", "correct"]:
            assert field in p, f"Missing field: {field}"


def test_batch_size_splits_correctly(tmp_path):
    """Ensure batching doesn't drop or duplicate samples."""
    records = make_records(tmp_path, n=7)
    for bs in [1, 3, 7, 100]:
        task = BinaryClassificationTask(batch_size=bs)
        result = task.run(DummyModel(), records, run_id="r1")
        assert len(result["predictions"]) == 7

import tempfile
import os
import pytest
from benchmarks.result_store import ResultStore


@pytest.fixture
def store(tmp_path):
    return ResultStore(db_path=str(tmp_path / "test.db"))


def test_save_and_retrieve_run(store):
    store.save_run(
        run_id="abc123",
        model_name="resnet50_timm",
        task="classification",
        dataset="cifar10",
        split="test",
        metrics={"top1_acc": 0.92, "total_samples": 100},
    )
    df = store.get_runs()
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "resnet50_timm"
    assert df.iloc[0]["metrics"]["top1_acc"] == 0.92


def test_accumulation(store):
    for i in range(3):
        store.save_run(
            run_id=f"run{i}",
            model_name="resnet50_timm",
            task="classification",
            dataset="cifar10",
            split="test",
            metrics={"top1_acc": 0.9 + i * 0.01},
        )
    df = store.get_runs()
    assert len(df) == 3


def test_save_predictions(store):
    preds = [
        {"run_id": "r1", "sample_id": "0", "image_path": "", "label": "cat", "pred": "cat", "confidence": 0.9, "correct": 1},
        {"run_id": "r1", "sample_id": "1", "image_path": "", "label": "dog", "pred": "cat", "confidence": 0.6, "correct": 0},
    ]
    store.save_predictions(preds)
    df = store.get_predictions("r1")
    assert len(df) == 2
    assert df["correct"].sum() == 1


def test_leaderboard(store):
    store.save_run("r1", "resnet50", "classification", "cifar10", "test", {"top1_acc": 0.91})
    store.save_run("r2", "vit", "classification", "cifar10", "test", {"top1_acc": 0.95})
    lb = store.leaderboard()
    assert lb.iloc[0]["model_name"] == "vit"

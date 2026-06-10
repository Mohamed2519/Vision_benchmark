"""
Integration test: seed the DB with synthetic results and verify the HTML report
is generated and contains expected content.
"""
import json
from pathlib import Path

import pytest

from benchmarks.result_store import ResultStore
from reports.generate_report import ReportBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_db(store: ResultStore):
    """Insert two models × two datasets with all required metric fields."""
    import random, math
    random.seed(42)

    models   = ["densenet121", "resnet50"]
    datasets = ["dataset1", "dataset2"]

    for model in models:
        for ds in datasets:
            rid = f"{model[:4]}-{ds[-1]}"
            auroc   = round(random.uniform(0.75, 0.97), 4)
            vt_auc  = round(auroc - random.uniform(0, 0.05), 4)
            g_auc   = round(auroc - random.uniform(0, 0.05), 4)
            metrics = {
                "combined_auroc": auroc,
                "val_test_auroc": vt_auc,
                "golden_auroc":   g_auc,
                "auroc":          auroc,
                "auprc":          round(auroc - 0.05, 4),
                "accuracy":       round(random.uniform(0.80, 0.95), 4),
                "sensitivity":    round(random.uniform(0.75, 0.95), 4),
                "specificity":    round(random.uniform(0.75, 0.95), 4),
                "precision":      round(random.uniform(0.75, 0.95), 4),
                "f1":             round(random.uniform(0.78, 0.94), 4),
                "youden_j":       round(random.uniform(0.55, 0.88), 4),
                "ece":            round(random.uniform(0.02, 0.12), 4),
                "tp": random.randint(50, 200),
                "tn": random.randint(50, 200),
                "fp": random.randint(5,  40),
                "fn": random.randint(5,  40),
                "total": 400,
                "threshold": 0.5,
            }
            store.save_run(
                run_id=rid,
                model_name=model,
                task="binary_classification",
                dataset=ds,
                split="combined",
                metrics=metrics,
            )
            # synthetic predictions for confidence distribution
            preds = []
            for i in range(40):
                correct = int(i % 5 != 0)
                preds.append({
                    "run_id":     rid,
                    "sample_id":  str(i),
                    "image_path": "",
                    "label":      i % 2,
                    "pred":       (i % 2) if correct else (1 - i % 2),
                    "confidence": round(0.9 if correct else 0.7, 3),
                    "correct":    correct,
                    "extra":      {},
                })
            store.save_predictions(preds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_builds(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "results.db"))
    _seed_db(store)

    builder = ReportBuilder(db_path=tmp_path / "results.db", title="Test Report")
    html = builder.build()

    assert isinstance(html, str)
    assert len(html) > 5000


def test_report_contains_model_names(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "results.db"))
    _seed_db(store)

    builder = ReportBuilder(db_path=tmp_path / "results.db", title="Test Report")
    html = builder.build()

    assert "densenet121" in html
    assert "resnet50"    in html
    assert "dataset1"    in html
    assert "dataset2"    in html


def test_report_contains_key_sections(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "results.db"))
    _seed_db(store)

    builder = ReportBuilder(db_path=tmp_path / "results.db", title="Test Report")
    html = builder.build()

    for section in ["Leaderboard", "Per-Model Detail", "AUROC Overview",
                    "Metrics Heatmap", "Calibration"]:
        assert section in html, f"Section '{section}' not found in HTML"


def test_report_contains_plotly(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "results.db"))
    _seed_db(store)

    builder = ReportBuilder(db_path=tmp_path / "results.db", title="Test Report")
    html = builder.build()

    assert "Plotly.newPlot" in html
    assert "plotly" in html.lower()


def test_report_table_sortable(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "results.db"))
    _seed_db(store)

    builder = ReportBuilder(db_path=tmp_path / "results.db", title="Test Report")
    html = builder.build()

    assert "sortTable" in html
    assert "Combined AUROC" in html


def test_empty_db_returns_no_results_page(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "empty.db"))
    builder = ReportBuilder(db_path=tmp_path / "empty.db", title="Empty")
    html = builder.build()
    assert "No results" in html


def test_report_written_to_file(tmp_path):
    store = ResultStore(db_path=str(tmp_path / "results.db"))
    _seed_db(store)

    out = tmp_path / "report.html"
    builder = ReportBuilder(db_path=tmp_path / "results.db", title="File Test")
    html = builder.build()
    out.write_text(html, encoding="utf-8")

    assert out.exists()
    assert out.stat().st_size > 10_000

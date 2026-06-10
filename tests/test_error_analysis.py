import pytest
from error_analysis.analyzer import ErrorAnalyzer


def make_preds(n_correct=8, n_wrong=2):
    preds = []
    for i in range(n_correct):
        preds.append({"run_id": "r1", "sample_id": str(i), "image_path": "",
                      "label": "cat", "pred": "cat", "confidence": 0.9, "correct": 1})
    for i in range(n_wrong):
        preds.append({"run_id": "r1", "sample_id": str(n_correct + i), "image_path": "",
                      "label": "dog", "pred": "cat", "confidence": 0.7, "correct": 0})
    return preds


def test_summary(tmp_path):
    preds = make_preds()
    analyzer = ErrorAnalyzer(preds, cfg={"top_n_failures": 5}, run_id="r1", output_dir=str(tmp_path))
    report = analyzer.analyze()
    assert report["summary"]["accuracy"] == 0.8
    assert report["summary"]["errors"] == 2


def test_confusion_pairs(tmp_path):
    preds = make_preds(n_wrong=3)
    analyzer = ErrorAnalyzer(preds, cfg={"top_n_failures": 5}, run_id="r1", output_dir=str(tmp_path))
    report = analyzer.analyze()
    assert report["confusion_pairs"][0]["true"] == "dog"
    assert report["confusion_pairs"][0]["pred"] == "cat"
    assert report["confusion_pairs"][0]["count"] == 3


def test_ece_range(tmp_path):
    preds = make_preds()
    analyzer = ErrorAnalyzer(preds, cfg={"top_n_failures": 5}, run_id="r1", output_dir=str(tmp_path))
    report = analyzer.analyze()
    assert 0.0 <= report["ece"] <= 1.0


def test_output_files_created(tmp_path):
    preds = make_preds()
    analyzer = ErrorAnalyzer(preds, cfg={"top_n_failures": 5}, run_id="r1", output_dir=str(tmp_path))
    analyzer.analyze()
    out = tmp_path / "error_analysis" / "r1"
    assert (out / "report.json").exists()
    assert (out / "per_class.csv").exists()
    assert (out / "top_failures.csv").exists()

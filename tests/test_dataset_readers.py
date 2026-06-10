"""
Tests for CSV and golden dataset readers using temporary files/folders.
"""
import csv
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from datasets.csv_reader import load_csv_split
from datasets.golden_reader import load_golden_split
from datasets.base import Split


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_image(path: Path):
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    img.save(str(path))


def _make_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Path", "label"])
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------
# CSV reader
# -----------------------------------------------------------------------

def test_csv_int_labels(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    imgs = [img_dir / f"img{i}.jpg" for i in range(4)]
    for p in imgs:
        _make_image(p)

    csv_file = tmp_path / "data.csv"
    _make_csv(csv_file, [
        {"Path": str(imgs[0]), "label": 0},
        {"Path": str(imgs[1]), "label": 1},
        {"Path": str(imgs[2]), "label": 0},
        {"Path": str(imgs[3]), "label": 1},
    ])

    records = load_csv_split(csv_file, dataset_name="test_ds")
    assert len(records) == 4
    assert records[0].label == 0
    assert records[1].label == 1
    assert all(r.split == Split.VAL_TEST for r in records)
    assert all(r.dataset_name == "test_ds" for r in records)


def test_csv_string_labels(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    imgs = [img_dir / f"img{i}.jpg" for i in range(3)]
    for p in imgs:
        _make_image(p)

    csv_file = tmp_path / "data.csv"
    _make_csv(csv_file, [
        {"Path": str(imgs[0]), "label": "normal"},
        {"Path": str(imgs[1]), "label": "Abnormal"},
        {"Path": str(imgs[2]), "label": "NORMAL"},
    ])

    records = load_csv_split(csv_file, dataset_name="test_ds")
    assert records[0].label == 0
    assert records[1].label == 1
    assert records[2].label == 0


def test_csv_merge_two_files(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    imgs = [img_dir / f"img{i}.jpg" for i in range(6)]
    for p in imgs:
        _make_image(p)

    val_csv  = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _make_csv(val_csv,  [{"Path": str(p), "label": 0} for p in imgs[:3]])
    _make_csv(test_csv, [{"Path": str(p), "label": 1} for p in imgs[3:]])

    records = load_csv_split([val_csv, test_csv], dataset_name="merged")
    assert len(records) == 6


# -----------------------------------------------------------------------
# Golden reader
# -----------------------------------------------------------------------

def _make_golden_set(tmp_path, samples):
    """samples: list of (stem, label_text)"""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    for stem, lbl_text in samples:
        _make_image(img_dir / f"{stem}.jpg")
        (ann_dir / f"{stem}.txt").write_text(lbl_text)

    return img_dir, ann_dir


def test_golden_convention_a_int(tmp_path):
    img_dir, ann_dir = _make_golden_set(tmp_path, [
        ("img0", "0"), ("img1", "1"), ("img2", "0"),
    ])
    records = load_golden_split(img_dir, ann_dir, "test_ds")
    assert len(records) == 3
    assert records[0].label == 0
    assert records[1].label == 1


def test_golden_convention_b_string(tmp_path):
    img_dir, ann_dir = _make_golden_set(tmp_path, [
        ("img0", "normal"), ("img1", "Abnormal"), ("img2", "NORMAL"),
    ])
    records = load_golden_split(img_dir, ann_dir, "test_ds")
    assert records[0].label == 0
    assert records[1].label == 1
    assert records[2].label == 0


def test_golden_convention_c_keyvalue(tmp_path):
    img_dir, ann_dir = _make_golden_set(tmp_path, [
        ("img0", "Finding: Normal"),
        ("img1", "class: abnormal"),
        ("img2", "label=normal"),
    ])
    records = load_golden_split(img_dir, ann_dir, "test_ds")
    assert records[0].label == 0
    assert records[1].label == 1
    assert records[2].label == 0


def test_golden_missing_annotation_warns(tmp_path):
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    img_dir.mkdir(); ann_dir.mkdir()

    _make_image(img_dir / "img_no_ann.jpg")
    _make_image(img_dir / "img_ok.jpg")
    (ann_dir / "img_ok.txt").write_text("1")

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        records = load_golden_split(img_dir, ann_dir, "test_ds")
        assert any("skipped" in str(x.message).lower() for x in w)

    assert len(records) == 1


def test_golden_split_tag(tmp_path):
    img_dir, ann_dir = _make_golden_set(tmp_path, [("img0", "0")])
    records = load_golden_split(img_dir, ann_dir, "myds")
    assert records[0].split == Split.GOLDEN
    assert records[0].dataset_name == "myds"

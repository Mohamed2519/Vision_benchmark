"""
Tests for PathologyClassifierModel wrapper.

Uses a tiny randomly-initialised in-memory model so no real checkpoint is needed.
"""
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

# Trigger registration of all pathology_* models
import models.chest_xray.pathology_classifier_model  # noqa: F401

from benchmarks.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path: Path):
    Image.new("RGB", (64, 64), color=128).save(str(path))


def _save_fake_checkpoint(path: Path, architecture: str = "densenet121"):
    """
    Save a minimal checkpoint that PathologyClassifier can load:
      - 'config'            : architecture / image_size / crop_size
      - 'model_state_dict'  : random weights matching the architecture
    """
    from torchvision import models as tvm

    if architecture == "densenet121":
        net = tvm.densenet121(weights=None)
        net.classifier = nn.Linear(net.classifier.in_features, 1)
    elif architecture == "resnet50":
        net = tvm.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 1)
    elif architecture == "efficientnet_b5":
        net = tvm.efficientnet_b5(weights=None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unsupported arch in test helper: {architecture}")

    torch.save(
        {
            "config": {
                "architecture": architecture,
                "image_size": 64,
                "crop_size": 64,
            },
            "model_state_dict": net.state_dict(),
        },
        str(path),
    )


# ---------------------------------------------------------------------------
# Registration tests (no checkpoint needed)
# ---------------------------------------------------------------------------

def test_all_architectures_registered():
    from models.chest_xray.pathology_classifier_model import _SUPPORTED_ARCHITECTURES
    registered = ModelRegistry.list_models()
    for arch in _SUPPORTED_ARCHITECTURES:
        assert f"pathology_{arch}" in registered, f"pathology_{arch} not registered"


def test_generic_name_registered():
    assert "pathology_classifier" in ModelRegistry.list_models()


# ---------------------------------------------------------------------------
# End-to-end inference tests (tiny random checkpoint, small images)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["densenet121", "resnet50", "efficientnet_b5"])
def test_predict_batch_output_format(tmp_path, arch):
    """predict_batch must return [{"score": float, "label": int}] per image."""
    ckpt = tmp_path / f"{arch}.pth"
    _save_fake_checkpoint(ckpt, arch)

    imgs = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        _make_image(p)
        imgs.append(str(p))

    model_cls = ModelRegistry.get(f"pathology_{arch}")
    model = model_cls(checkpoint_path=str(ckpt), device="cpu")

    results = model.predict_batch(imgs)

    assert len(results) == 3
    for r in results:
        assert "score" in r,  "missing 'score'"
        assert "label" in r,  "missing 'label'"
        assert 0.0 <= r["score"] <= 1.0, f"score out of range: {r['score']}"
        assert r["label"] in (0, 1), f"label not binary: {r['label']}"


def test_label_matches_threshold(tmp_path):
    """label must equal int(score >= threshold)."""
    ckpt = tmp_path / "dn.pth"
    _save_fake_checkpoint(ckpt, "densenet121")

    img = tmp_path / "img.jpg"
    _make_image(img)

    model_cls = ModelRegistry.get("pathology_densenet121")
    model = model_cls(checkpoint_path=str(ckpt), device="cpu", threshold=0.5)
    results = model.predict_batch([str(img)])

    r = results[0]
    expected_label = int(r["score"] >= 0.5)
    assert r["label"] == expected_label


def test_generic_registration_loads(tmp_path):
    """The architecture-agnostic 'pathology_classifier' name should work."""
    ckpt = tmp_path / "dn.pth"
    _save_fake_checkpoint(ckpt, "densenet121")

    img = tmp_path / "img.jpg"
    _make_image(img)

    model_cls = ModelRegistry.get("pathology_classifier")
    model = model_cls(checkpoint_path=str(ckpt), device="cpu")
    results = model.predict_batch([str(img)])

    assert len(results) == 1
    assert "score" in results[0]


def test_missing_checkpoint_raises():
    model_cls = ModelRegistry.get("pathology_densenet121")
    with pytest.raises(FileNotFoundError):
        model_cls(checkpoint_path="/nonexistent/path.pth")

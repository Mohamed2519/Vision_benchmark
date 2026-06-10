"""
PathologyClassifier wrapper for the benchmark framework.

Wraps the self-contained PathologyClassifier (inference.py) so that any
checkpoint trained with the matching training pipeline can be plugged
into the evaluator with zero changes to either side.

Supported architectures (read from checkpoint config automatically):
    densenet121, resnet50,
    efficientnet_b5, efficientnet_b6, efficientnet_b7,
    vgg16,
    vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
    swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b,
    convnext_tiny, convnext_small, convnext_base, convnext_large

Registration names follow the pattern  "pathology_<architecture>", e.g.:
    "pathology_efficientnet_b5"
    "pathology_densenet121"
    "pathology_swin_b"

A generic "pathology_classifier" name is also registered for cases where
the architecture is embedded in the checkpoint and you don't want to
specify it up front.

Usage in configs / CLI
----------------------
    python evaluate.py run -m pathology_efficientnet_b5 -d dataset1 \\
        --checkpoint checkpoints/efficientnet_b5_best.pth

Or programmatically:

    from benchmarks.registry import ModelRegistry
    cls = ModelRegistry.get("pathology_efficientnet_b5")
    model = cls(checkpoint_path="checkpoints/best.pth", device="cuda")
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Make sure the package root is on the path when this file is imported
# directly or from tests outside the package.
_pkg_root = Path(__file__).parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from benchmarks.registry import ModelRegistry
from models.chest_xray.base_model import ChestXrayModel

# PathologyClassifier is imported lazily inside _PathologyAdapter.__init__
# so that importing this module (for registration) does not require torch.
_PathologyClassifier = None


def _get_classifier():
    global _PathologyClassifier
    if _PathologyClassifier is None:
        from models.chest_xray.inference import PathologyClassifier
        _PathologyClassifier = PathologyClassifier
    return _PathologyClassifier


# Architectures supported by PathologyClassifier — duplicated here so
# registration can happen at import time without pulling in torch.
_SUPPORTED_ARCHITECTURES = {
    "densenet121", "resnet50",
    "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "vgg16",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
    "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
}


# ---------------------------------------------------------------------------
# Shared adapter logic
# ---------------------------------------------------------------------------

class _PathologyAdapter(ChestXrayModel):
    """
    Thin adapter that bridges PathologyClassifier → ChestXrayModel interface.

    PathologyClassifier.predict_batch() returns:
        [{"probability": float, "prediction": int, "confidence": float}, ...]

    ChestXrayModel.predict_batch() must return:
        [{"score": float,        "label": int}, ...]

    score = P(abnormal) = probability
    label = hard binary prediction

    Parameters
    ----------
    checkpoint_path : path to the .pth checkpoint
    device          : 'cuda' | 'cpu' | None (auto-detect)
    threshold       : decision boundary for the hard label (default 0.5)
    architecture    : override checkpoint architecture (optional)
    image_size      : override checkpoint image size (optional)
    crop_size       : override checkpoint crop size (optional)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        architecture: Optional[str] = None,
        image_size: Optional[int] = None,
        crop_size: Optional[int] = None,
    ):
        super().__init__(device=device or "cpu")
        self._clf = _get_classifier()(
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=threshold,
            architecture=architecture,
            image_size=image_size,
            crop_size=crop_size,
        )

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        Run inference on a list of image paths.

        Returns
        -------
        List of {"score": P(abnormal), "label": 0 or 1}
        """
        raw = self._clf.predict_batch(image_paths)
        return [
            {
                "score": r["probability"],   # P(abnormal) ∈ [0, 1]
                "label": r["prediction"],    # hard binary label
            }
            for r in raw
        ]

    def set_threshold(self, threshold: float) -> None:
        self._clf.set_threshold(threshold)

    @property
    def name(self) -> str:
        return f"pathology_{self._clf.architecture}"


# ---------------------------------------------------------------------------
# Register once per supported architecture so users can reference models by
# architecture name without passing architecture= kwarg every time.
# ---------------------------------------------------------------------------

def _make_arch_class(arch: str):
    """Return a new subclass pre-configured for *arch*."""

    class _ArchModel(_PathologyAdapter):
        def __init__(
            self,
            checkpoint_path: str,
            device: Optional[str] = None,
            threshold: float = 0.5,
            image_size: Optional[int] = None,
            crop_size: Optional[int] = None,
        ):
            super().__init__(
                checkpoint_path=checkpoint_path,
                device=device,
                threshold=threshold,
                architecture=arch,
                image_size=image_size,
                crop_size=crop_size,
            )

        @property
        def name(self) -> str:
            return f"pathology_{arch}"

    _ArchModel.__name__ = f"Pathology_{arch}"
    return _ArchModel


for _arch in _SUPPORTED_ARCHITECTURES:
    ModelRegistry.register(f"pathology_{_arch}")(_make_arch_class(_arch))


# Generic registration: architecture is read from the checkpoint at load time.
@ModelRegistry.register("pathology_classifier")
class PathologyClassifierModel(_PathologyAdapter):
    """
    Architecture-agnostic wrapper — reads architecture from the checkpoint.

    Use this when you don't want to specify the architecture in the
    model name, or when evaluating multiple checkpoints of different
    architectures via the same registered name.
    """
    pass

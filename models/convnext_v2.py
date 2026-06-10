"""
CONVNEXT_V2 — ConvNeXt-Base trained on unified_data_v1 (unified_v3clean split).

Checkpoint
----------
    /mnt/nvme/echonova-vision/dev/pathology_binary_classifier/checkpoints/
    unified_data_v1/convnext_base_unified_data_v1_unified_v3clean/best_model.pth

Registration name : "CONVNEXT_V2"
Architecture      : convnext_base
"""
from __future__ import annotations

from typing import List, Optional

from benchmarks.registry import ModelRegistry
from models.base import ChestXrayModel

CHECKPOINT_PATH = (
    "/mnt/nvme/echonova-vision/dev/pathology_binary_classifier/checkpoints/"
    "unified_data_v1/convnext_base_unified_data_v1_unified_v3clean/best_model.pth"
)


@ModelRegistry.register("CONVNEXT_V2")
class ConvNextV2(ChestXrayModel):
    """
    ConvNeXt-Base binary chest X-ray classifier (unified_data_v1 / v3clean).

    Parameters
    ----------
    checkpoint_path : override the default checkpoint path
    device          : 'cuda' | 'cpu' | None (auto-detect)
    threshold       : decision boundary for the hard label (default 0.5)
    """

    def __init__(
        self,
        checkpoint_path: str = CHECKPOINT_PATH,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        super().__init__(device=device or "cpu")
        from models.inference import PathologyClassifier

        self._clf = PathologyClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=threshold,
            architecture="convnext_base",
        )

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        results = self._clf.predict_batch(image_paths)
        return [{"score": r["probability"], "label": r["prediction"]} for r in results]

    @property
    def name(self) -> str:
        return "CONVNEXT_V2"

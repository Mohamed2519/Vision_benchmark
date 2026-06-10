"""
SWIN_B — Swin Transformer-B trained on vinDr + CheXpert + NIH + PadChest + MIMIC.

Checkpoint
----------
    /mnt/nvme/echonova-vision/dev/pathology_binary_classifier/checkpoints/
    vinDr-chexpert-nih-padchest-mimic/swin_b_vinDr-chexpert-nih-padchest-mimic/best_model.pth

Registration name : "SWIN_B"
Architecture      : swin_b
"""
from __future__ import annotations

from typing import List, Optional

from benchmarks.registry import ModelRegistry
from models.base import ChestXrayModel

CHECKPOINT_PATH = (
    "/mnt/nvme/echonova-vision/dev/pathology_binary_classifier/checkpoints/"
    "vinDr-chexpert-nih-padchest-mimic/"
    "swin_b_vinDr-chexpert-nih-padchest-mimic/best_model.pth"
)


@ModelRegistry.register("SWIN_B")
class SwinB(ChestXrayModel):
    """
    Swin Transformer-B binary chest X-ray classifier
    (vinDr + CheXpert + NIH + PadChest + MIMIC).

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
            architecture="swin_b",
        )

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        results = self._clf.predict_batch(image_paths)
        return [{"score": r["probability"], "label": r["prediction"]} for r in results]

    @property
    def name(self) -> str:
        return "SWIN_B"

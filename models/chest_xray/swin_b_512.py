"""
SWIN_B_512 — Swin Transformer-B (512 px) trained on vinDr + CheXpert + NIH + PadChest + MIMIC.

Checkpoint
----------
    /mnt/nvme/echonova-vision/dev/pathology_binary_classifier/checkpoints/
    vinDr-chexpert-nih-padchest-mimic/swin_b_vinDr-chexpert-nih-padchest-mimic_512/best_model.pth

Registration name : "SWIN_B_512"
Architecture      : swin_b
Image / crop size : 512 × 512
"""
from __future__ import annotations

from typing import List, Optional

from benchmarks.registry import ModelRegistry
from models.chest_xray.base_model import ChestXrayModel

CHECKPOINT_PATH = (
    "/mnt/nvme/echonova-vision/dev/pathology_binary_classifier/checkpoints/"
    "vinDr-chexpert-nih-padchest-mimic/"
    "swin_b_vinDr-chexpert-nih-padchest-mimic_512/best_model.pth"
)


@ModelRegistry.register("SWIN_B_512")
class SwinB512(ChestXrayModel):
    """
    Swin Transformer-B (512 px input) binary chest X-ray classifier
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

        from models.chest_xray.inference import PathologyClassifier

        self._clf = PathologyClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=threshold,
            architecture="swin_b",
            image_size=512,
            crop_size=512,
        )

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        results = self._clf.predict_batch(image_paths)
        return [{"score": r["probability"], "label": r["prediction"]} for r in results]

    @property
    def name(self) -> str:
        return "SWIN_B_512"

"""
Base class for all chest X-ray binary classification models.

Every model must implement predict_batch().
predict_batch receives a list of image file paths (strings) and must return
a list of dicts:

    [
        {"label": 0, "score": 0.12},   # normal,   score = P(abnormal)
        {"label": 1, "score": 0.87},   # abnormal
        ...
    ]

score is always the probability of the ABNORMAL class (class 1).
label is the hard decision (0 or 1); if omitted, the evaluator derives it
from score >= threshold.

How to add a new model
----------------------
1. Create a file in  models/chest_xray/  (e.g. my_model.py)
2. Subclass ChestXrayModel
3. Implement __init__() to load weights and predict_batch()
4. Decorate with @ModelRegistry.register("my_model_name")
5. Done — no other changes needed.

Example skeleton
----------------
from benchmarks.registry import ModelRegistry
from models.chest_xray.base_model import ChestXrayModel

@ModelRegistry.register("my_resnet_chest")
class MyResNetChest(ChestXrayModel):
    def __init__(self, weights_path: str, device: str = "cpu"):
        super().__init__(device=device)
        self.model = ...load your model...

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        images = [self.load_image(p) for p in image_paths]
        scores = ...run inference and return P(abnormal)...
        return [{"score": float(s), "label": int(s >= 0.5)} for s in scores]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from PIL import Image


class ChestXrayModel(ABC):
    """
    Minimal interface for a binary chest X-ray classification model.

    Parameters
    ----------
    device : 'cpu', 'cuda', or 'mps'
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        Run inference on a batch of image file paths.

        Parameters
        ----------
        image_paths : list of absolute paths to image files

        Returns
        -------
        list of dicts, one per image:
            score : float  — probability of ABNORMAL class, in [0, 1]
            label : int    — hard prediction (0=normal, 1=abnormal)
                             optional; derived from score >= 0.5 if absent
        """

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> Image.Image:
        """Load and convert to RGB PIL Image."""
        return Image.open(path).convert("RGB")

    def load_images(self, paths: List[str]) -> List[Image.Image]:
        return [self.load_image(p) for p in paths]

    @property
    def name(self) -> str:
        return self.__class__.__name__

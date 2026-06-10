"""
Base class every chest X-ray model must implement.

To add a new model
------------------
1. Create a file in  models/  (e.g. my_model.py)
2. Subclass ChestXrayModel and implement predict_batch()
3. Decorate the class with @ModelRegistry.register("MY_MODEL")
4. Done — the runner auto-discovers all files in models/

predict_batch contract
----------------------
Input  : List[str]  — absolute paths to image files
Output : List[dict] — one dict per image:
    score : float  — probability of ABNORMAL class, in [0, 1]
    label : int    — hard prediction (0 = normal, 1 = abnormal)
                     may be omitted; evaluator derives it from score >= 0.5
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from PIL import Image


class ChestXrayModel(ABC):
    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """Return [{"score": float, "label": int}, ...] for each image path."""

    def load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @property
    def name(self) -> str:
        return self.__class__.__name__

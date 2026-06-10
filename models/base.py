"""
Base class every vision model must implement.

To add a new model:
1. Create a file in models/ (e.g. models/my_model.py)
2. Subclass BaseVisionModel and implement predict()
3. Decorate the class with @ModelRegistry.register("my_model")
4. That's it — the runner will auto-discover it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image


class BaseVisionModel(ABC):
    """
    Minimal interface every benchmarked model must satisfy.

    Parameters
    ----------
    device : str
        'cuda', 'cpu', or 'mps'
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def predict(self, images: List[Image.Image]) -> List[dict]:
        """
        Run inference on a batch of PIL images.

        Returns
        -------
        list of dict, one per image.
        Each dict must have at minimum:
            label  : str   — top predicted class name
            conf   : float — confidence in [0, 1]
            probs  : dict  — {class_name: probability, ...}  (optional but recommended)
        """

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Optional preprocessing hook. Override to customise."""
        return image

    @property
    def name(self) -> str:
        return self.__class__.__name__

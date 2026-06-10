"""
Example model: fine-tuned DenseNet-121 for chest X-ray binary classification.

Replace the weight loading and architecture with your actual model.
The key contract is predict_batch() returning {"score": ..., "label": ...}.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models, transforms

from benchmarks.registry import ModelRegistry
from models.chest_xray.base_model import ChestXrayModel


@ModelRegistry.register("densenet121_chest")
class DenseNet121Chest(ChestXrayModel):
    """
    DenseNet-121 with a binary output head.
    score = sigmoid(logit)  →  P(abnormal)
    """

    def __init__(
        self,
        weights_path: str = "",
        device: str = "cpu",
        image_size: int = 224,
    ):
        super().__init__(device=device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        net = models.densenet121(weights=None)
        net.classifier = nn.Linear(net.classifier.in_features, 1)

        if weights_path:
            state = torch.load(weights_path, map_location=device)
            # support both raw state_dict and checkpoint dicts
            if "state_dict" in state:
                state = state["state_dict"]
            net.load_state_dict(state, strict=False)

        self.model = net.to(device).eval()

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        images = self.load_images(image_paths)
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)

        with torch.no_grad():
            logits = self.model(tensors).squeeze(1)   # (B,)
            scores = torch.sigmoid(logits).cpu().numpy()

        return [
            {"score": float(s), "label": int(s >= 0.5)}
            for s in scores
        ]

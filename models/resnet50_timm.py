"""
Example model: ResNet-50 via timm.

Shows the minimal boilerplate needed to plug any timm model into the benchmark.
"""
from __future__ import annotations

from typing import List

import torch
from PIL import Image

from benchmarks.registry import ModelRegistry
from models.base import BaseVisionModel


@ModelRegistry.register("resnet50_timm")
class ResNet50Timm(BaseVisionModel):
    def __init__(self, device: str = "cpu", pretrained: bool = True):
        super().__init__(device=device)
        import timm

        self.model = timm.create_model("resnet50", pretrained=pretrained)
        self.model.eval().to(device)

        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.class_names = self._load_imagenet_classes()

    def predict(self, images: List[Image.Image]) -> List[dict]:
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors)
            probs = torch.softmax(logits, dim=-1)

        results = []
        for prob in probs:
            top_idx = prob.argmax().item()
            results.append(
                {
                    "label": self.class_names[top_idx],
                    "conf": prob[top_idx].item(),
                    "probs": {self.class_names[i]: prob[i].item() for i in prob.topk(5).indices},
                }
            )
        return results

    @staticmethod
    def _load_imagenet_classes() -> list[str]:
        try:
            import timm
            from timm.data import ImageNetInfo
            return ImageNetInfo().index_to_description()
        except Exception:
            return [str(i) for i in range(1000)]

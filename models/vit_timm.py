"""
Example model: ViT-Base/16 via timm.
"""
from __future__ import annotations

from typing import List

import torch
from PIL import Image

from benchmarks.registry import ModelRegistry
from models.base import BaseVisionModel


@ModelRegistry.register("vit_base_timm")
class ViTBaseTimm(BaseVisionModel):
    def __init__(self, device: str = "cpu", pretrained: bool = True, model_name: str = "vit_base_patch16_224"):
        super().__init__(device=device)
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained)
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
            from timm.data import ImageNetInfo
            return ImageNetInfo().index_to_description()
        except Exception:
            return [str(i) for i in range(1000)]

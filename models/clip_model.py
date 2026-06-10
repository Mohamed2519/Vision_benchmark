"""
Example model: OpenAI CLIP via HuggingFace transformers.
Supports zero-shot classification with custom class names.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image

from benchmarks.registry import ModelRegistry
from models.base import BaseVisionModel


@ModelRegistry.register("clip_vit_b32")
class CLIPViTB32(BaseVisionModel):
    def __init__(
        self,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
        prompt_template: str = "a photo of a {}",
    ):
        super().__init__(device=device)
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.model.eval()
        self.class_names = class_names or []
        self.prompt_template = prompt_template
        self._text_features = None

    def set_classes(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self._text_features = None  # reset cache

    def _get_text_features(self):
        if self._text_features is None:
            texts = [self.prompt_template.format(c) for c in self.class_names]
            inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                self._text_features = self.model.get_text_features(**inputs)
                self._text_features /= self._text_features.norm(dim=-1, keepdim=True)
        return self._text_features

    def predict(self, images: List[Image.Image]) -> List[dict]:
        if not self.class_names:
            raise RuntimeError("CLIP requires class_names. Call set_classes() first.")

        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        text_features = self._get_text_features()
        with torch.no_grad():
            img_features = self.model.get_image_features(**inputs)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_features @ text_features.T).softmax(dim=-1)

        results = []
        for prob in logits:
            top_idx = prob.argmax().item()
            results.append(
                {
                    "label": self.class_names[top_idx],
                    "conf": prob[top_idx].item(),
                    "probs": {self.class_names[i]: prob[i].item() for i in prob.topk(min(5, len(self.class_names))).indices},
                }
            )
        return results

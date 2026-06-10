"""
Classification task runner.

Supports any dataset loadable via torchvision or HuggingFace datasets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm


class ClassificationTask:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size", 32)
        self.top_k = cfg.get("top_k", [1, 5])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(self, model, dataset: str, split: str, run_id: str) -> Dict:
        images, labels, paths = self._load_dataset(dataset, split, model)

        # Set classes for zero-shot models (e.g. CLIP)
        if hasattr(model, "set_classes"):
            unique_labels = sorted(set(labels))
            model.set_classes(unique_labels)

        predictions = []
        label_list = list(dict.fromkeys(labels))  # preserve order, unique

        for i in tqdm(range(0, len(images), self.batch_size), desc="Inference"):
            batch_imgs = images[i : i + self.batch_size]
            batch_labels = labels[i : i + self.batch_size]
            batch_paths = paths[i : i + self.batch_size]

            preds = model.predict(batch_imgs)
            for j, (pred, gt) in enumerate(zip(preds, batch_labels)):
                predictions.append(
                    {
                        "run_id": run_id,
                        "sample_id": str(i + j),
                        "image_path": batch_paths[j],
                        "label": gt,
                        "pred": pred["label"],
                        "confidence": pred["conf"],
                        "correct": int(pred["label"] == gt),
                        "extra": {"top5": list(pred.get("probs", {}).keys())},
                    }
                )

        metrics = self._compute_metrics(predictions)
        return {"metrics": metrics, "predictions": predictions}

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_dataset(self, dataset: str, split: str, model):
        """
        Supports:
        - HuggingFace dataset hub IDs  e.g. 'cifar10', 'food101'
        - Local folders with ImageFolder layout  e.g. '/data/my_dataset'
        """
        if Path(dataset).exists():
            return self._load_imagefolder(dataset, split)
        else:
            return self._load_hf_dataset(dataset, split)

    def _load_hf_dataset(self, name: str, split: str):
        from datasets import load_dataset

        ds = load_dataset(name, split=split, trust_remote_code=True)
        img_col = self._detect_image_col(ds.column_names)
        lbl_col = self._detect_label_col(ds.column_names)
        label_names = ds.features[lbl_col].names if hasattr(ds.features[lbl_col], "names") else None

        images, labels, paths = [], [], []
        for row in ds:
            img = row[img_col]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
            images.append(img)
            lbl = row[lbl_col]
            labels.append(label_names[lbl] if label_names else str(lbl))
            paths.append("")
        return images, labels, paths

    def _load_imagefolder(self, root: str, split: str):
        from torchvision.datasets import ImageFolder

        path = Path(root) / split
        if not path.exists():
            path = Path(root)

        ds = ImageFolder(str(path))
        images, labels, paths = [], [], []
        for img_path, class_idx in ds.imgs:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            labels.append(ds.classes[class_idx])
            paths.append(img_path)
        return images, labels, paths

    @staticmethod
    def _detect_image_col(cols):
        for candidate in ("image", "img", "pixel_values"):
            if candidate in cols:
                return candidate
        raise ValueError(f"Cannot detect image column in: {cols}")

    @staticmethod
    def _detect_label_col(cols):
        for candidate in ("label", "labels", "fine_label", "coarse_label"):
            if candidate in cols:
                return candidate
        raise ValueError(f"Cannot detect label column in: {cols}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        correct = sum(p["correct"] for p in predictions)
        total = len(predictions)
        metrics: Dict[str, float] = {"top1_acc": correct / total if total else 0.0}

        # top-k accuracy (needs probs in extra)
        for k in self.top_k:
            if k == 1:
                continue
            topk_correct = sum(
                1
                for p in predictions
                if p["label"] in (p.get("extra") or {}).get("top5", [])[:k]
            )
            metrics[f"top{k}_acc"] = topk_correct / total if total else 0.0

        metrics["total_samples"] = total
        metrics["correct_samples"] = correct
        return metrics

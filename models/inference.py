"""
Self-contained inference engine for chest X-ray pathology classification.

This module provides a production-ready inference pipeline for single and batch
inference from trained binary classifier checkpoints. Handles preprocessing and
model loading automatically.

Usage:
    >>> classifier = PathologyClassifier(
    ...     checkpoint_path="path/to/best_model.pth",
    ...     architecture="efficientnet_b5"
    ... )
    >>> result = classifier.predict_image("path/to/image.jpg")
    >>> results = classifier.predict_batch(["img1.jpg", "img2.jpg"])
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional


class PathologyClassifier:
    """Binary classifier for chest X-ray pathology detection.

    Supports both single image and batch inference with automatic preprocessing.
    Loads checkpoints trained with train_util.create_model() and automatically
    reads all configuration (architecture, image size, etc.) from the checkpoint.

    Args:
        checkpoint_path: Path to trained model checkpoint (.pth file).
                        All configuration is loaded from the checkpoint.
        device: Device to run inference on ('cuda', 'cpu', or None for auto).
                If None, auto-detects GPU availability.
        threshold: Classification threshold for binary predictions (default: 0.5).
                  Can be adjusted with set_threshold().
        architecture: (Optional) Override checkpoint architecture. If provided,
                     this takes precedence over checkpoint config.
        image_size: (Optional) Override checkpoint image size.
        crop_size: (Optional) Override checkpoint crop size.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint missing required config or unsupported architecture
        KeyError: If checkpoint missing model weights

    Example:
        >>> classifier = PathologyClassifier("checkpoints/model.pth")
        >>> result = classifier.predict_image("image.jpg")
    """

    # Preprocessing configuration: uniform normalization for grayscale X-rays
    # Using average of ImageNet RGB channels for consistent grayscale handling
    MEAN = [0.449, 0.449, 0.449]
    STD = [0.226, 0.226, 0.226]

    SUPPORTED_ARCHITECTURES = {
        "densenet121", "resnet50",
        "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
        "vgg16",
        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
        "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b",
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    }

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        architecture: Optional[str] = None,
        image_size: Optional[int] = None,
        crop_size: Optional[int] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set defaults
        self.architecture = "efficientnet_b5"
        self.image_size = 512
        self.crop_size = 512

        # Try to load configuration from checkpoint (new format)
        config_found = self._load_config_from_checkpoint()

        # If config not in checkpoint, try to infer architecture from checkpoint path
        if not config_found and architecture is None:
            inferred_arch = self._infer_architecture_from_path()
            if inferred_arch:
                self.architecture = inferred_arch

        # Override with provided arguments (highest priority)
        if architecture is not None:
            self.architecture = architecture.lower()
        if image_size is not None:
            self.image_size = image_size
        if crop_size is not None:
            self.crop_size = crop_size

        # Validate architecture
        if self.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{self.architecture}'. "
                f"Choose from: {sorted(self.SUPPORTED_ARCHITECTURES)}"
            )

        # Load model and checkpoint
        self.model = self._build_model()
        self._load_checkpoint()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Build preprocessing pipeline
        self.transform = self._build_transform()

    def _load_config_from_checkpoint(self) -> bool:
        """Load and extract configuration from checkpoint file.

        Returns:
            True if config was found in checkpoint, False otherwise.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # Try to load config from checkpoint (new format)
        if "config" in checkpoint:
            config = checkpoint["config"]
            # Extract required fields with defaults
            self.architecture = config.get("architecture", "efficientnet_b5").lower()
            self.image_size = config.get("image_size", 512)
            self.crop_size = config.get("crop_size", 512)

            print(f"✓ Config loaded from checkpoint:")
            print(f"  Architecture: {self.architecture}")
            print(f"  Image size: {self.image_size}")
            print(f"  Crop size: {self.crop_size}")
            return True
        else:
            # Old checkpoint format (no config)
            return False

    def _infer_architecture_from_path(self) -> Optional[str]:
        """Try to infer architecture from checkpoint file path.

        Returns:
            Architecture name if found in path, None otherwise.
        """
        checkpoint_str = str(self.checkpoint_path).lower()

        for arch in self.SUPPORTED_ARCHITECTURES:
            if arch in checkpoint_str:
                return arch

        return None

    def _build_model(self) -> nn.Module:
        """Build model architecture with binary classification head."""
        if self.architecture == "densenet121":
            model = models.densenet121(weights=None)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 1)

        elif self.architecture == "resnet50":
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 1)

        elif self.architecture in ["efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]:
            if self.architecture == "efficientnet_b5":
                model = models.efficientnet_b5(weights=None)
            elif self.architecture == "efficientnet_b6":
                model = models.efficientnet_b6(weights=None)
            else:
                model = models.efficientnet_b7(weights=None)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 1)

        elif self.architecture == "vgg16":
            model = models.vgg16(weights=None)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 1)

        # Vision Transformer variants
        elif self.architecture in ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]:
            if self.architecture == "vit_b_16":
                model = models.vit_b_16(weights=None)
            elif self.architecture == "vit_b_32":
                model = models.vit_b_32(weights=None)
            elif self.architecture == "vit_l_16":
                model = models.vit_l_16(weights=None)
            elif self.architecture == "vit_l_32":
                model = models.vit_l_32(weights=None)
            else:
                model = models.vit_h_14(weights=None)
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, 1)

        # Swin Transformer variants
        elif self.architecture in ["swin_t", "swin_s", "swin_b"]:
            if self.architecture == "swin_t":
                model = models.swin_t(weights=None)
            elif self.architecture == "swin_s":
                model = models.swin_s(weights=None)
            else:
                model = models.swin_b(weights=None)
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, 1)

        # Swin Transformer V2 variants
        elif self.architecture in ["swin_v2_t", "swin_v2_s", "swin_v2_b"]:
            if self.architecture == "swin_v2_t":
                model = models.swin_v2_t(weights=None)
            elif self.architecture == "swin_v2_s":
                model = models.swin_v2_s(weights=None)
            else:
                model = models.swin_v2_b(weights=None)
            num_features = model.head.in_features
            model.head = nn.Linear(num_features, 1)

        # ConvNeXt variants
        elif self.architecture in ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]:
            if self.architecture == "convnext_tiny":
                model = models.convnext_tiny(weights=None)
            elif self.architecture == "convnext_small":
                model = models.convnext_small(weights=None)
            elif self.architecture == "convnext_base":
                model = models.convnext_base(weights=None)
            else:
                model = models.convnext_large(weights=None)
            num_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(num_features, 1)

        return model

    def _load_checkpoint(self) -> None:
        """Load model weights from checkpoint file."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load model state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback: checkpoint might be just state dict
            self.model.load_state_dict(checkpoint)

        print(f"✓ Model weights loaded from checkpoint")
        print(f"  Device: {self.device}")

    def _build_transform(self) -> transforms.Compose:
        """Build image preprocessing pipeline (inference mode)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image and convert to RGB."""
        img = Image.open(image_path)
        # Convert grayscale or RGBA to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @torch.no_grad()
    def predict_image(
        self,
        image_path: Union[str, Path],
        return_logit: bool = False,
    ) -> Dict[str, Union[float, int, bool]]:
        """
        Predict pathology class for a single image.

        Args:
            image_path: Path to input image (jpg, png, etc.)
            return_logit: If True, include raw logit in output

        Returns:
            Dictionary with keys:
                - 'probability': float in [0, 1], probability of abnormality
                - 'prediction': int (0=normal, 1=abnormal)
                - 'confidence': float, max(probability, 1-probability)
                - 'logit': float (optional, if return_logit=True)

        Raises:
            FileNotFoundError: If image file not found
            RuntimeError: If image processing fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess
        img = self._load_image(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Forward pass
        logit = self.model(img_tensor).squeeze()
        prob = torch.sigmoid(logit).cpu().item()
        pred = int(prob >= self.threshold)

        result = {
            "probability": prob,
            "prediction": pred,
            "confidence": max(prob, 1.0 - prob),
        }

        if return_logit:
            result["logit"] = float(logit.cpu().item())

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        return_logits: bool = False,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        """
        Predict pathology class for multiple images.

        Args:
            image_paths: List of paths to input images
            batch_size: Number of images per batch
            return_logits: If True, include raw logits in output

        Returns:
            List of prediction dictionaries (same format as predict_image)

        Raises:
            ValueError: If image_paths is empty
            FileNotFoundError: If any image file not found
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        # Verify all images exist
        image_paths = [Path(p) for p in image_paths]
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

        all_results = []

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            # Load and stack images
            imgs = []
            for path in batch_paths:
                img = self._load_image(path)
                img_tensor = self.transform(img)
                imgs.append(img_tensor)

            imgs_batch = torch.stack(imgs).to(self.device)

            # Forward pass
            logits = self.model(imgs_batch).squeeze()

            # Handle single image case (logits becomes scalar)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            logits_np = logits.cpu().numpy()

            # Build results
            for prob, logit in zip(probs, logits_np):
                pred = int(prob >= self.threshold)
                result = {
                    "probability": float(prob),
                    "prediction": pred,
                    "confidence": float(max(prob, 1.0 - prob)),
                }
                if return_logits:
                    result["logit"] = float(logit)
                all_results.append(result)

        return all_results

    def predict_batch_with_paths(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        return_logits: bool = False,
    ) -> List[Dict[str, Union[str, float, int, bool]]]:
        """
        Predict with image paths included in results.

        Args:
            image_paths: List of paths to input images
            batch_size: Number of images per batch
            return_logits: If True, include raw logits in output

        Returns:
            List of prediction dictionaries with 'path' field added
        """
        results = self.predict_batch(image_paths, batch_size, return_logits)

        for path, result in zip(image_paths, results):
            result["path"] = str(path)

        return results

    def set_threshold(self, threshold: float) -> None:
        """
        Update classification threshold.

        Args:
            threshold: Probability threshold for positive class (0 to 1)

        Raises:
            ValueError: If threshold not in [0, 1]
        """
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

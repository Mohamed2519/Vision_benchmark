"""
DatasetLoader — central registry of all 5 datasets.

Configure each dataset in  configs/datasets.yaml  and call:

    loader = DatasetLoader("configs/datasets.yaml")
    records = loader.load("dataset1", Split.GOLDEN)
    all_records = loader.load_all()        # all datasets, both splits
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .base import SampleRecord, Split
from .csv_reader import load_csv_split
from .golden_reader import load_golden_split


class DatasetLoader:
    def __init__(self, config_path: str = "configs/datasets.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self._datasets: Dict[str, dict] = cfg["datasets"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dataset_names(self) -> List[str]:
        return list(self._datasets.keys())

    def load(
        self,
        dataset_name: str,
        split: Split,
    ) -> List[SampleRecord]:
        """Load one (dataset, split) combination."""
        dcfg = self._datasets[dataset_name]

        if split == Split.VAL_TEST:
            return self._load_val_test(dataset_name, dcfg)
        elif split == Split.GOLDEN:
            return self._load_golden(dataset_name, dcfg)
        else:
            raise ValueError(f"Unknown split: {split}")

    def load_dataset(self, dataset_name: str) -> Dict[Split, List[SampleRecord]]:
        """Load both splits for one dataset."""
        return {
            Split.VAL_TEST: self.load(dataset_name, Split.VAL_TEST),
            Split.GOLDEN:   self.load(dataset_name, Split.GOLDEN),
        }

    def load_all(self) -> Dict[str, Dict[Split, List[SampleRecord]]]:
        """Load all datasets, all splits."""
        return {name: self.load_dataset(name) for name in self._datasets}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_val_test(self, name: str, dcfg: dict) -> List[SampleRecord]:
        if "val_test" not in dcfg:
            raise KeyError(
                f"Dataset '{name}' is missing the required 'val_test' key in datasets.yaml. "
                f"Found keys: {list(dcfg.keys())}"
            )
        vt = dcfg["val_test"]
        csv_paths = vt["csv"] if isinstance(vt["csv"], list) else [vt["csv"]]
        image_root = vt.get("image_root")
        return load_csv_split(
            csv_paths=csv_paths,
            dataset_name=name,
            image_root=image_root,
        )

    def _load_golden(self, name: str, dcfg: dict) -> List[SampleRecord]:
        g = dcfg.get("golden", {})
        images_dir      = g.get("images_dir")
        annotations_dir = g.get("annotations_dir")

        # Gracefully return empty list when no golden set is configured.
        if not images_dir or not annotations_dir:
            import warnings
            warnings.warn(
                f"[{name}] No golden set configured (images_dir or annotations_dir is null). "
                "Skipping golden split — combined AUROC will be computed from val+test only."
            )
            return []

        return load_golden_split(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            dataset_name=name,
        )

"""Shared types for the chest X-ray dataset layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Split(str, Enum):
    VAL_TEST = "val_test"   # merged val + test, sourced from CSV
    GOLDEN   = "golden"     # golden set, sourced from folder + annotations


@dataclass
class SampleRecord:
    """One labelled image sample."""
    image_path: Path
    label: int              # 0 = normal, 1 = abnormal
    dataset_name: str
    split: Split
    sample_id: str = ""

    def __post_init__(self):
        self.image_path = Path(self.image_path)
        if not self.sample_id:
            self.sample_id = self.image_path.stem

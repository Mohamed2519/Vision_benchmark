"""
Reads the golden set: folder of images + folder of .txt annotation files.

Annotation .txt format (auto-detected, three common conventions):

  Convention A — single integer per file:
      0
      (or 1)

  Convention B — label on its own line (case-insensitive):
      normal
      (or abnormal)

  Convention C — key:value or key=value anywhere in the file:
      class: normal
      label=abnormal
      Finding: Abnormal

The reader tries all three and raises a clear error if none match.

Image ↔ annotation pairing:
  - Same stem: chest_001.jpg  ↔  chest_001.txt
  - Annotation file for image NOT found → warning, sample skipped.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from .base import SampleRecord, Split


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_LABEL_MAP = {
    "0": 0, "normal": 0, "neg": 0, "negative": 0,
    "1": 1, "abnormal": 1, "pos": 1, "positive": 1,
}


def _parse_txt(txt_path: Path) -> int:
    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()

    # Convention A / B — single token
    first_token = text.split()[0].lower().rstrip(".,;") if text.split() else ""
    if first_token in _LABEL_MAP:
        return _LABEL_MAP[first_token]

    # Convention C — key: value anywhere in the file
    for line in text.splitlines():
        line = line.strip().lower()
        m = re.search(r"(?:class|label|finding|category)\s*[=:]\s*(\w+)", line)
        if m:
            token = m.group(1).rstrip(".,;")
            if token in _LABEL_MAP:
                return _LABEL_MAP[token]

    raise ValueError(
        f"Cannot parse label from '{txt_path}'. "
        f"File content: {text[:200]!r}"
    )


def load_golden_split(
    images_dir: str | Path,
    annotations_dir: str | Path,
    dataset_name: str,
    image_extensions: Optional[set] = None,
) -> List[SampleRecord]:
    """
    Load the golden set from two folders.

    Parameters
    ----------
    images_dir       : folder containing the X-ray images
    annotations_dir  : folder containing the .txt annotation files
    dataset_name     : human-readable dataset tag
    image_extensions : override the default supported image extensions
    """
    images_dir      = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    exts = image_extensions or _IMAGE_EXTS

    # Build annotation index: stem -> Path
    ann_index: Dict[str, Path] = {
        p.stem: p
        for p in annotations_dir.iterdir()
        if p.suffix.lower() == ".txt"
    }

    records: List[SampleRecord] = []
    missing = 0

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue

        ann_path = ann_index.get(img_path.stem)
        if ann_path is None:
            warnings.warn(
                f"[{dataset_name}/golden] No annotation for '{img_path.name}' — skipped."
            )
            missing += 1
            continue

        try:
            label = _parse_txt(ann_path)
        except ValueError as e:
            warnings.warn(str(e) + " — skipped.")
            missing += 1
            continue

        records.append(
            SampleRecord(
                image_path=img_path,
                label=label,
                dataset_name=dataset_name,
                split=Split.GOLDEN,
            )
        )

    if missing:
        warnings.warn(
            f"[{dataset_name}/golden] {missing} image(s) skipped (missing or unreadable annotation)."
        )

    return records

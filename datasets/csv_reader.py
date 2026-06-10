"""
Reads the val+test CSV files.

Expected CSV columns (case-insensitive):
    Path  — absolute or relative path to the image file
    label — integer 0 (normal) or 1 (abnormal)
              also accepts string 'normal'/'abnormal' and variants

Multiple CSVs (val.csv + test.csv) can be merged by passing a list.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Union

from .base import SampleRecord, Split


# Map string labels to int
_LABEL_MAP = {
    "normal": 0, "neg": 0, "negative": 0, "0": 0, 0: 0,
    "abnormal": 1, "pos": 1, "positive": 1, "1": 1, 1: 1,
}


def _parse_label(raw) -> int:
    key = str(raw).strip().lower()
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    raise ValueError(f"Unrecognised label value: '{raw}'. Expected 0/1 or normal/abnormal.")


def _find_col(columns: List[str], candidates: List[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    raise KeyError(f"Cannot find any of {candidates} in columns: {columns}")


def load_csv_split(
    csv_paths: Union[str, Path, List[Union[str, Path]]],
    dataset_name: str,
    image_root: Union[str, Path, None] = None,
) -> List[SampleRecord]:
    """
    Load one or more CSV files and return a flat list of SampleRecords.

    Parameters
    ----------
    csv_paths   : path or list of paths to CSV files (val.csv, test.csv …)
    dataset_name: human-readable dataset tag used in results
    image_root  : optional prefix prepended to relative image paths in the CSV
    """
    if not isinstance(csv_paths, list):
        csv_paths = [csv_paths]

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates()

    path_col  = _find_col(df.columns.tolist(), ["path", "image_path", "filepath", "file_path", "filename"])
    label_col = _find_col(df.columns.tolist(), ["label", "class", "target", "y"])

    records: List[SampleRecord] = []
    for _, row in df.iterrows():
        img_path = Path(str(row[path_col]).strip())
        if image_root and not img_path.is_absolute():
            img_path = Path(image_root) / img_path

        records.append(
            SampleRecord(
                image_path=img_path,
                label=_parse_label(row[label_col]),
                dataset_name=dataset_name,
                split=Split.VAL_TEST,
            )
        )
    return records

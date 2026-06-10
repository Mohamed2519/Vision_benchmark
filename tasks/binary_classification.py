"""
Binary classification task runner for chest X-ray Normal / Abnormal.

This runner is completely decoupled from any specific model.
It accepts any callable that implements the ModelProtocol interface.

Flow
----
1. Receive a list of SampleRecords (already loaded by DatasetLoader)
2. Call model.predict_batch() on batches of image paths
3. Collect (y_true, y_pred, y_score) triples
4. Compute binary metrics
5. Return structured results for persistence + W&B logging
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from datasets.base import SampleRecord, Split
from tasks.binary_metrics import compute_metrics


class BinaryClassificationTask:
    """
    Parameters
    ----------
    batch_size  : images per model call
    threshold   : score → label decision boundary (default 0.5)
    """

    def __init__(self, batch_size: int = 16, threshold: float = 0.5):
        self.batch_size = batch_size
        self.threshold  = threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model,                          # any object with predict_batch()
        records: List[SampleRecord],
        run_id: str,
        desc: str = "Inference",
    ) -> Dict[str, Any]:
        """
        Run inference over *records* and return a result dict.

        Returns
        -------
        {
            "metrics":     {...},
            "predictions": [ {run_id, sample_id, image_path, label,
                               pred, confidence, correct, ...} ]
        }
        """
        y_true:   List[int]   = []
        y_pred:   List[int]   = []
        y_score:  List[float] = []
        raw_preds: List[Dict] = []

        for batch_records in self._batches(records):
            paths = [str(r.image_path) for r in batch_records]

            # ---- model call ----
            batch_results = model.predict_batch(paths)
            # batch_results: List[{"label": int, "score": float}]
            # score = probability of class 1 (abnormal)

            for rec, res in zip(batch_records, batch_results):
                score = float(res["score"])
                pred  = int(res.get("label", int(score >= self.threshold)))

                y_true.append(rec.label)
                y_pred.append(pred)
                y_score.append(score)

                raw_preds.append({
                    "run_id":     run_id,
                    "sample_id":  rec.sample_id,
                    "image_path": str(rec.image_path),
                    "label":      rec.label,
                    "pred":       pred,
                    "confidence": score,
                    "correct":    int(pred == rec.label),
                    "extra": {
                        "dataset": rec.dataset_name,
                        "split":   rec.split.value,
                    },
                })

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            threshold=self.threshold,
        )

        return {"metrics": metrics, "predictions": raw_preds}

    # ------------------------------------------------------------------
    def _batches(self, records: List[SampleRecord]):
        for i in range(0, len(records), self.batch_size):
            yield records[i : i + self.batch_size]

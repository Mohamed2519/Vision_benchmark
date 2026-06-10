"""W&B integration for benchmark runs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class WandbLogger:
    def __init__(self, project: str, entity: Optional[str], run_name: str):
        import wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            reinit=True,
        )

    def log(
        self,
        model_name: str,
        task: str,
        dataset: str,
        split: str,
        metrics: Dict[str, Any],
        predictions: List[Dict],
        error_report: Dict,
        cfg: Dict,
    ) -> None:
        wandb = self._wandb

        # Flat metrics
        self._run.log({f"metrics/{k}": v for k, v in metrics.items()})
        self._run.config.update(
            {"model": model_name, "task": task, "dataset": dataset, "split": split}
        )

        # Confusion matrix (classification)
        if task == "classification" and predictions:
            labels = [p["label"] for p in predictions]
            preds = [p["pred"] for p in predictions]
            classes = sorted(set(labels) | set(preds))
            if cfg.get("log_confusion_matrix"):
                self._run.log(
                    {
                        "confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=labels,
                            preds=preds,
                            class_names=classes,
                        )
                    }
                )

        # Sample failure images
        if cfg.get("log_images") and predictions:
            failures = [p for p in predictions if not p.get("correct")]
            max_imgs = cfg.get("max_images_logged", 50)
            logged = []
            for p in failures[:max_imgs]:
                img_path = p.get("image_path")
                if img_path:
                    logged.append(
                        wandb.Image(
                            img_path,
                            caption=f"true={p['label']} pred={p['pred']} conf={p.get('confidence', 0):.2f}",
                        )
                    )
            if logged:
                self._run.log({"failure_images": logged})

        # Error analysis summary
        if error_report:
            self._run.log(
                {f"error_analysis/{k}": v for k, v in error_report.get("summary", {}).items()}
            )

    def finish(self) -> None:
        self._run.finish()

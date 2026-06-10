"""W&B integration — supports both general classification and binary medical tasks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


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

    # ------------------------------------------------------------------
    # Binary classification (chest X-ray)
    # ------------------------------------------------------------------

    def log_binary(
        self,
        model_name: str,
        dataset: str,
        vt_metrics: Dict[str, Any],
        g_metrics: Dict[str, Any],
        combined_metrics: Dict[str, Any],
        predictions: List[Dict],
        error_report: Dict,
        cfg: Dict,
    ) -> None:
        wandb = self._wandb
        run   = self._run

        run.config.update({"model": model_name, "dataset": dataset})

        # Per-split metrics
        for split_tag, m in [("val_test", vt_metrics), ("golden", g_metrics), ("combined", combined_metrics)]:
            run.log({f"{split_tag}/{k}": v for k, v in m.items() if isinstance(v, (int, float))})

        # ROC curve
        if predictions:
            y_true  = [p["label"]      for p in predictions]
            y_score = [p["confidence"] for p in predictions]
            run.log({"roc_curve": wandb.plot.roc_curve(y_true, [[1 - s, s] for s in y_score], labels=["normal", "abnormal"])})
            run.log({"pr_curve":  wandb.plot.pr_curve(y_true,  [[1 - s, s] for s in y_score], labels=["normal", "abnormal"])})

            # Confusion matrix
            if cfg.get("log_confusion_matrix", True):
                y_pred = [p["pred"] for p in predictions]
                run.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=["normal", "abnormal"],
                    )
                })

            # Failure images
            if cfg.get("log_images", True):
                max_imgs = cfg.get("max_images_logged", 30)
                failures = [p for p in predictions if not p.get("correct")]
                logged   = []
                for p in failures[:max_imgs]:
                    img_path = p.get("image_path")
                    if img_path:
                        try:
                            logged.append(
                                wandb.Image(
                                    img_path,
                                    caption=(
                                        f"true={'abnormal' if p['label']==1 else 'normal'} "
                                        f"pred={'abnormal' if p['pred']==1 else 'normal'} "
                                        f"score={p.get('confidence', 0):.3f} "
                                        f"split={p.get('extra', {}).get('split', '')}"
                                    ),
                                )
                            )
                        except Exception:
                            pass
                if logged:
                    run.log({"failure_images": logged})

        # Error analysis summary
        if error_report:
            run.log({f"error_analysis/{k}": v for k, v in error_report.get("summary", {}).items()
                     if isinstance(v, (int, float))})

    # ------------------------------------------------------------------
    # Generic classification (kept for backward compat)
    # ------------------------------------------------------------------

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
        self._run.log({f"metrics/{k}": v for k, v in metrics.items()})
        self._run.config.update({"model": model_name, "task": task, "dataset": dataset, "split": split})

        if task == "classification" and predictions:
            labels  = [p["label"] for p in predictions]
            preds_l = [p["pred"]  for p in predictions]
            classes = sorted(set(labels) | set(preds_l))
            if cfg.get("log_confusion_matrix"):
                self._run.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None, y_true=labels, preds=preds_l, class_names=classes,
                    )
                })

        if error_report:
            self._run.log({f"error_analysis/{k}": v for k, v in error_report.get("summary", {}).items()})

    def finish(self) -> None:
        self._run.finish()

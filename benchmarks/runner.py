"""
BenchmarkRunner — orchestrates a full benchmark trial for one model.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.table import Table

from benchmarks.registry import ModelRegistry
from benchmarks.result_store import ResultStore
from benchmarks.wandb_logger import WandbLogger
from error_analysis.analyzer import ErrorAnalyzer
from tasks.classification import ClassificationTask

console = Console()

TASK_MAP = {
    "classification": ClassificationTask,
}


class BenchmarkRunner:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.store = ResultStore(
            str(Path(self.cfg["benchmark"]["output_dir"]) / "results.db")
        )

    def run(
        self,
        model_name: str,
        dataset: str,
        split: str = "test",
        task: str = "classification",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())[:8]
        model_kwargs = model_kwargs or {}

        console.rule(f"[bold cyan]Benchmark: {model_name} | {task} | {dataset}/{split}")

        # --- Build model ---
        ModelRegistry.autodiscover()
        model_cls = ModelRegistry.get(model_name)
        model = model_cls(**model_kwargs)

        # --- Run task ---
        task_cfg = self.cfg["tasks"].get(task, {})
        task_runner = TASK_MAP[task](cfg=task_cfg)
        results = task_runner.run(model=model, dataset=dataset, split=split, run_id=run_id)

        metrics = results["metrics"]
        predictions = results["predictions"]

        # --- Persist ---
        self.store.save_run(
            run_id=run_id,
            model_name=model_name,
            task=task,
            dataset=dataset,
            split=split,
            metrics=metrics,
            config={**self.cfg, "model_kwargs": model_kwargs},
        )
        self.store.save_predictions(predictions)
        self.store.export_csv()

        # --- Error analysis ---
        if self.cfg["error_analysis"]["enabled"]:
            analyzer = ErrorAnalyzer(
                predictions=predictions,
                cfg=self.cfg["error_analysis"],
                run_id=run_id,
                output_dir=self.cfg["benchmark"]["output_dir"],
            )
            error_report = analyzer.analyze()
        else:
            error_report = {}

        # --- W&B logging ---
        wandb_cfg = self.cfg.get("wandb", {})
        if wandb_cfg.get("enabled"):
            logger = WandbLogger(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                run_name=f"{model_name}-{run_id}",
            )
            logger.log(
                model_name=model_name,
                task=task,
                dataset=dataset,
                split=split,
                metrics=metrics,
                predictions=predictions,
                error_report=error_report,
                cfg=wandb_cfg,
            )
            logger.finish()

        # --- Print summary ---
        self._print_summary(model_name, task, dataset, split, metrics)

        return {"run_id": run_id, "metrics": metrics, "error_report": error_report}

    def _print_summary(self, model, task, dataset, split, metrics):
        table = Table(title=f"Results — {model} on {dataset}/{split} ({task})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)

    def leaderboard(self, task: str = "classification", metric: str = "top1_acc"):
        df = self.store.leaderboard(task=task, metric=metric)
        table = Table(title=f"Leaderboard — {task} ({metric})")
        for col in df.columns:
            table.add_column(col, style="cyan" if col == metric else "white")
        for _, row in df.iterrows():
            table.add_row(*[str(v) for v in row])
        console.print(table)
        return df

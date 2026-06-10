"""
Medical Chest X-Ray Binary Classification Evaluator
=====================================================

Runs every registered model against every configured dataset (both splits),
persists accumulated results, generates error reports, logs to W&B,
and prints a ranked leaderboard.

Usage
-----
# Evaluate all models on all datasets
python evaluate.py run-all

# Evaluate specific models
python evaluate.py run --models densenet121_chest resnet50_chest --datasets dataset1 dataset2

# Print leaderboard from stored results
python evaluate.py leaderboard

# Re-run error analysis on a past run
python evaluate.py error-report <run_id>
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import click
import yaml
from rich.console import Console
from rich.table import Table

from benchmarks.registry import ModelRegistry
from benchmarks.result_store import ResultStore
from benchmarks.wandb_logger import WandbLogger
from datasets.base import Split
from datasets.loader import DatasetLoader
from error_analysis.analyzer import ErrorAnalyzer
from tasks.binary_classification import BinaryClassificationTask
from tasks.binary_metrics import combine_scores

console = Console()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

@click.group()
@click.option("--config",   default="configs/default.yaml",  show_default=True)
@click.option("--datasets-config", default="configs/datasets.yaml", show_default=True)
@click.pass_context
def cli(ctx, config, datasets_config):
    ctx.ensure_object(dict)
    ctx.obj["config"]          = config
    ctx.obj["datasets_config"] = datasets_config


@cli.command("run")
@click.option("--models",   "-m", multiple=True, required=True,
              help="Registered model name(s). Repeat flag for multiple.")
@click.option("--datasets", "-d", multiple=True, default=(),
              help="Dataset name(s). Default: all configured datasets.")
@click.option("--device",   default="cpu", show_default=True)
@click.option("--batch-size", default=16, show_default=True, type=int)
@click.option("--threshold", default=0.5, show_default=True, type=float)
@click.pass_context
def run_cmd(ctx, models, datasets, device, batch_size, threshold):
    """Evaluate one or more models on one or more datasets."""
    evaluator = Evaluator(
        config_path=ctx.obj["config"],
        datasets_config=ctx.obj["datasets_config"],
    )
    ds_names = list(datasets) or evaluator.loader.dataset_names()

    for model_name in models:
        for dataset_name in ds_names:
            evaluator.run_single(
                model_name=model_name,
                dataset_name=dataset_name,
                device=device,
                batch_size=batch_size,
                threshold=threshold,
            )


@cli.command("run-all")
@click.option("--device",     default="cpu",  show_default=True)
@click.option("--batch-size", default=16,     show_default=True, type=int)
@click.option("--threshold",  default=0.5,    show_default=True, type=float)
@click.pass_context
def run_all(ctx, device, batch_size, threshold):
    """Evaluate ALL registered models on ALL configured datasets."""
    evaluator = Evaluator(
        config_path=ctx.obj["config"],
        datasets_config=ctx.obj["datasets_config"],
    )
    ModelRegistry.autodiscover(package="models.chest_xray")
    model_names   = ModelRegistry.list_models()
    dataset_names = evaluator.loader.dataset_names()

    console.print(f"Models  : {model_names}")
    console.print(f"Datasets: {dataset_names}")

    for model_name in model_names:
        for dataset_name in dataset_names:
            evaluator.run_single(
                model_name=model_name,
                dataset_name=dataset_name,
                device=device,
                batch_size=batch_size,
                threshold=threshold,
            )


@cli.command()
@click.option("--sort-by", default="combined_auroc", show_default=True)
@click.pass_context
def leaderboard(ctx, sort_by):
    """Print accumulated leaderboard sorted by combined AUROC."""
    evaluator = Evaluator(
        config_path=ctx.obj["config"],
        datasets_config=ctx.obj["datasets_config"],
    )
    evaluator.print_leaderboard(sort_by=sort_by)


@cli.command("report")
@click.option("--out", default="results/benchmark_report.html", show_default=True,
              help="Output HTML file path.")
@click.option("--title", default="Chest X-Ray Benchmark Report", show_default=True)
@click.pass_context
def report_cmd(ctx, out, title):
    """Generate an interactive HTML benchmark report."""
    import subprocess, sys
    subprocess.run([
        sys.executable, "reports/generate_report.py",
        "--db", f"{yaml.safe_load(open(ctx.obj['config']))['benchmark']['output_dir']}/results.db",
        "--out", out,
        "--title", title,
    ], check=True)
    console.print(f"[bold green]Report ready → {out}[/]")


@cli.command("error-report")
@click.argument("run_id")
@click.pass_context
def error_report(ctx, run_id):
    """Re-run error analysis for a saved run_id."""
    with open(ctx.obj["config"]) as f:
        cfg = yaml.safe_load(f)
    store = ResultStore(f"{cfg['benchmark']['output_dir']}/results.db")
    preds = store.get_predictions(run_id).to_dict("records")
    if not preds:
        console.print(f"[red]No predictions found for run_id={run_id}[/]")
        return
    analyzer = ErrorAnalyzer(
        predictions=preds,
        cfg=cfg["error_analysis"],
        run_id=run_id,
        output_dir=cfg["benchmark"]["output_dir"],
    )
    analyzer.analyze()


# -----------------------------------------------------------------------
# Core Evaluator class
# -----------------------------------------------------------------------

class Evaluator:
    def __init__(self, config_path: str, datasets_config: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.loader = DatasetLoader(datasets_config)
        self.store  = ResultStore(
            str(Path(self.cfg["benchmark"]["output_dir"]) / "results.db")
        )

    # ------------------------------------------------------------------
    def run_single(
        self,
        model_name: str,
        dataset_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        threshold: float = 0.5,
        model_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Full evaluation of one model × one dataset.

        Steps
        -----
        1. Load model
        2. Run inference on val_test split
        3. Run inference on golden split
        4. Compute combined AUROC
        5. Persist all results
        6. Error analysis
        7. W&B logging
        """
        run_id = str(uuid.uuid4())[:8]
        model_kwargs = model_kwargs or {"device": device}

        console.rule(f"[bold cyan]{model_name}  ×  {dataset_name}  [{run_id}]")

        # --- Load model ---
        ModelRegistry.autodiscover(package="models.chest_xray")
        model_cls = ModelRegistry.get(model_name)
        model     = model_cls(**model_kwargs)

        task = BinaryClassificationTask(batch_size=batch_size, threshold=threshold)

        # --- val_test split ---
        vt_records = self.loader.load(dataset_name, Split.VAL_TEST)
        console.print(f"  val_test : {len(vt_records)} samples")
        vt_results = task.run(model, vt_records, run_id=run_id + "_vt", desc="val+test")

        # --- golden split ---
        g_records  = self.loader.load(dataset_name, Split.GOLDEN)
        console.print(f"  golden   : {len(g_records)} samples")
        g_results  = task.run(model, g_records, run_id=run_id + "_g", desc="golden")

        # --- combined AUROC ---
        vt_pairs = [
            (p["label"], p["confidence"])
            for p in vt_results["predictions"]
        ]
        g_pairs  = [
            (p["label"], p["confidence"])
            for p in g_results["predictions"]
        ]
        combined_auroc = combine_scores(vt_pairs, g_pairs)

        # --- persist ---
        for split_tag, results in [("val_test", vt_results), ("golden", g_results)]:
            split_run_id = run_id + ("_vt" if split_tag == "val_test" else "_g")
            metrics_with_combined = {**results["metrics"], "combined_auroc": combined_auroc}

            self.store.save_run(
                run_id=split_run_id,
                model_name=model_name,
                task="binary_classification",
                dataset=dataset_name,
                split=split_tag,
                metrics=metrics_with_combined,
                config={"threshold": threshold, "batch_size": batch_size},
            )
            self.store.save_predictions(results["predictions"])

        # --- summary run (both splits combined) for the leaderboard ---
        all_preds = vt_results["predictions"] + g_results["predictions"]
        all_true  = [p["label"]      for p in all_preds]
        all_pred  = [p["pred"]       for p in all_preds]
        all_score = [p["confidence"] for p in all_preds]

        from tasks.binary_metrics import compute_metrics as cm
        combined_metrics = {
            **cm(all_true, all_pred, all_score, threshold=threshold),
            "combined_auroc":     combined_auroc,
            "val_test_auroc":     vt_results["metrics"].get("auroc"),
            "golden_auroc":       g_results["metrics"].get("auroc"),
            "val_test_samples":   len(vt_records),
            "golden_samples":     len(g_records),
        }
        self.store.save_run(
            run_id=run_id,
            model_name=model_name,
            task="binary_classification",
            dataset=dataset_name,
            split="combined",
            metrics=combined_metrics,
        )
        self.store.save_predictions([{**p, "run_id": run_id} for p in all_preds])
        self.store.export_csv()

        # --- error analysis ---
        if self.cfg["error_analysis"]["enabled"]:
            analyzer = ErrorAnalyzer(
                predictions=all_preds,
                cfg=self.cfg["error_analysis"],
                run_id=run_id,
                output_dir=self.cfg["benchmark"]["output_dir"],
            )
            error_report_data = analyzer.analyze()
        else:
            error_report_data = {}

        # --- W&B ---
        wandb_cfg = self.cfg.get("wandb", {})
        if wandb_cfg.get("enabled"):
            try:
                logger = WandbLogger(
                    project=wandb_cfg["project"],
                    entity=wandb_cfg.get("entity"),
                    run_name=f"{model_name}-{dataset_name}-{run_id}",
                )
                logger.log_binary(
                    model_name=model_name,
                    dataset=dataset_name,
                    vt_metrics=vt_results["metrics"],
                    g_metrics=g_results["metrics"],
                    combined_metrics=combined_metrics,
                    predictions=all_preds,
                    error_report=error_report_data,
                    cfg=wandb_cfg,
                )
                logger.finish()
            except Exception as e:
                console.print(f"[yellow]W&B logging failed: {e}[/]")

        # --- print ---
        self._print_split_table(model_name, dataset_name, vt_results["metrics"], g_results["metrics"], combined_auroc)

        return {
            "run_id":          run_id,
            "combined_auroc":  combined_auroc,
            "val_test_metrics": vt_results["metrics"],
            "golden_metrics":   g_results["metrics"],
            "combined_metrics": combined_metrics,
        }

    # ------------------------------------------------------------------
    def print_leaderboard(self, sort_by: str = "combined_auroc"):
        df = self.store.get_runs()
        if df.empty:
            console.print("[yellow]No results yet.[/]")
            return

        # Keep only combined rows
        df = df[df["split"] == "combined"].copy()
        if df.empty:
            console.print("[yellow]No combined results yet.[/]")
            return

        # Flatten metrics
        import pandas as pd
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df[["model_name", "dataset", "timestamp"]], metrics_df], axis=1)

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)

        cols = [
            "model_name", "dataset",
            "combined_auroc", "val_test_auroc", "golden_auroc",
            "accuracy", "sensitivity", "specificity", "f1",
            "auroc", "auprc", "timestamp",
        ]
        cols = [c for c in cols if c in df.columns]

        table = Table(title=f"Leaderboard — Chest X-Ray Binary Classification (sorted by {sort_by})")
        for col in cols:
            style = "bold green" if col == sort_by else ("cyan" if col in ("model_name", "dataset") else "white")
            table.add_column(col, style=style)

        for _, row in df.iterrows():
            def fmt(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v) if v is not None else "—"
            table.add_row(*[fmt(row.get(c)) for c in cols])

        console.print(table)
        return df

    # ------------------------------------------------------------------
    def _print_split_table(self, model, dataset, vt_m, g_m, combined_auroc):
        table = Table(title=f"{model}  ×  {dataset}")
        table.add_column("Metric",        style="cyan")
        table.add_column("val+test",      style="green")
        table.add_column("golden",        style="yellow")

        key_metrics = [
            "accuracy", "sensitivity", "specificity", "precision",
            "f1", "youden_j", "auroc", "auprc",
            "tp", "tn", "fp", "fn", "total",
        ]
        for k in key_metrics:
            vt_v = vt_m.get(k)
            g_v  = g_m.get(k)
            fmt  = lambda v: f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "—")
            table.add_row(k, fmt(vt_v), fmt(g_v))

        console.print(table)
        console.print(f"  [bold magenta]Combined AUROC (val+test+golden): {combined_auroc:.4f}[/]\n")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    cli()

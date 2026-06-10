#!/usr/bin/env python3
"""
Vision Benchmark CLI  —  single entry point for chest X-ray binary classification.

Usage examples
--------------
# Evaluate all registered models on all configured datasets (GPU by default)
python benchmark.py run-all

# Evaluate specific models on a specific dataset
python benchmark.py run --models CONVNEXT_V2 SWIN_B --datasets chexpert

# Print leaderboard
python benchmark.py leaderboard

# Generate HTML report
python benchmark.py report

# Re-run error analysis on a past run
python benchmark.py error-report <run_id>

# List registered models
python benchmark.py list-models
"""
import click
from rich.console import Console

console = Console()


@click.group()
@click.option("--config", default="configs/default.yaml", show_default=True,
              help="Path to config YAML")
@click.option("--datasets-config", default="configs/datasets.yaml", show_default=True,
              help="Path to datasets YAML")
@click.pass_context
def cli(ctx, config, datasets_config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["datasets_config"] = datasets_config


@cli.command("run")
@click.option("--models",    "-m", multiple=True, required=False,
              help="Registered model name(s). Default: all registered models.")
@click.option("--datasets",  "-d", multiple=True, default=(),
              help="Dataset name(s). Default: all configured datasets.")
@click.option("--device",    default="cuda", show_default=True)
@click.option("--batch-size", default=16,   show_default=True, type=int)
@click.option("--threshold",  default=0.5,  show_default=True, type=float)
@click.pass_context
def run_cmd(ctx, models, datasets, device, batch_size, threshold):
    """Evaluate one or more models on one or more datasets."""
    from benchmarks.registry import ModelRegistry
    from evaluate import Evaluator

    evaluator = Evaluator(
        config_path=ctx.obj["config"],
        datasets_config=ctx.obj["datasets_config"],
    )

    if not models:
        ModelRegistry.autodiscover(package="models")
        models = ModelRegistry.list_models()
        if not models:
            raise click.ClickException("No registered models found.")

    ds_names = list(datasets) or evaluator.loader.dataset_names()
    if not ds_names:
        raise click.ClickException("No datasets found.")

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
@click.option("--device",     default="cuda", show_default=True)
@click.option("--batch-size", default=16,     show_default=True, type=int)
@click.option("--threshold",  default=0.5,    show_default=True, type=float)
@click.pass_context
def run_all(ctx, device, batch_size, threshold):
    """Evaluate ALL registered models on ALL configured datasets."""
    from benchmarks.registry import ModelRegistry
    from evaluate import Evaluator

    evaluator = Evaluator(
        config_path=ctx.obj["config"],
        datasets_config=ctx.obj["datasets_config"],
    )
    ModelRegistry.autodiscover(package="models")
    model_names = ModelRegistry.list_models()
    if not model_names:
        raise click.ClickException("No registered models found.")

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
    from evaluate import Evaluator

    evaluator = Evaluator(
        config_path=ctx.obj["config"],
        datasets_config=ctx.obj["datasets_config"],
    )
    evaluator.print_leaderboard(sort_by=sort_by)


@cli.command("report")
@click.option("--out", default="results/benchmark_report.html", show_default=True)
@click.option("--title", default="Chest X-Ray Benchmark Report", show_default=True)
@click.pass_context
def report_cmd(ctx, out, title):
    """Generate an interactive HTML benchmark report."""
    import subprocess, sys, yaml

    cfg = yaml.safe_load(open(ctx.obj["config"]))
    db  = f"{cfg['benchmark']['output_dir']}/results.db"
    subprocess.run(
        [sys.executable, "reports/generate_report.py",
         "--db", db, "--out", out, "--title", title],
        check=True,
    )
    console.print(f"[bold green]Report ready → {out}[/]")


@cli.command("error-report")
@click.argument("run_id")
@click.pass_context
def error_report(ctx, run_id):
    """Re-run error analysis on a saved run."""
    import yaml
    from benchmarks.result_store import ResultStore
    from error_analysis.analyzer import ErrorAnalyzer

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


@cli.command("list-models")
def list_models():
    """List all registered models."""
    from benchmarks.registry import ModelRegistry

    ModelRegistry.autodiscover(package="models")
    models = ModelRegistry.list_models()
    console.print("[bold cyan]Registered models:[/]")
    for m in models:
        console.print(f"  • {m}")


if __name__ == "__main__":
    cli()

#!/usr/bin/env python3
"""
Vision Benchmark CLI

Usage examples
--------------
# Run a single model
python benchmark.py run resnet50_timm --dataset cifar10 --split test

# Run multiple models in sequence
python benchmark.py run resnet50_timm vit_base_timm --dataset cifar10

# Show the leaderboard
python benchmark.py leaderboard

# List registered models
python benchmark.py list-models
"""
import click
from rich.console import Console

console = Console()


@click.group()
@click.option("--config", default="configs/default.yaml", show_default=True, help="Path to config YAML")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--dataset", required=True, help="HuggingFace dataset ID or local ImageFolder path")
@click.option("--split", default="test", show_default=True)
@click.option("--task", default="classification", show_default=True, type=click.Choice(["classification"]))
@click.option("--device", default="cpu", show_default=True)
@click.pass_context
def run(ctx, models, dataset, split, task, device):
    """Run benchmark for one or more models."""
    from benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner(config_path=ctx.obj["config"])
    for model_name in models:
        runner.run(
            model_name=model_name,
            dataset=dataset,
            split=split,
            task=task,
            model_kwargs={"device": device},
        )


@cli.command()
@click.option("--task", default="classification", show_default=True)
@click.option("--metric", default="top1_acc", show_default=True)
@click.pass_context
def leaderboard(ctx, task, metric):
    """Print accumulated results leaderboard."""
    from benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner(config_path=ctx.obj["config"])
    runner.leaderboard(task=task, metric=metric)


@cli.command("list-models")
def list_models():
    """List all registered models."""
    from benchmarks.registry import ModelRegistry

    ModelRegistry.autodiscover()
    models = ModelRegistry.list_models()
    console.print("[bold cyan]Registered models:[/]")
    for m in models:
        console.print(f"  • {m}")


@cli.command("error-report")
@click.argument("run_id")
@click.option("--config", default="configs/default.yaml", show_default=True)
def error_report(run_id, config):
    """Re-run error analysis on a saved run."""
    import yaml
    from benchmarks.result_store import ResultStore
    from error_analysis.analyzer import ErrorAnalyzer

    with open(config) as f:
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


if __name__ == "__main__":
    cli()

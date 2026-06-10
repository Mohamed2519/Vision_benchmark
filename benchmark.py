#!/usr/bin/env python3
"""
Vision Benchmark CLI

Usage examples
--------------
# Run all registered models on all configured datasets (GPU by default)
python benchmark.py run

# Run a single model on one dataset
python benchmark.py run resnet50_timm --dataset cifar10 --split test

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
@click.option("--datasets-config", default="configs/datasets.yaml", show_default=True, help="Path to datasets YAML")
@click.pass_context
def cli(ctx, config, datasets_config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["datasets_config"] = datasets_config


def _resolve_models(models):
    if models:
        return list(models)

    from benchmarks.registry import ModelRegistry

    ModelRegistry.autodiscover()
    model_names = ModelRegistry.list_models()
    if not model_names:
        raise click.ClickException("No registered models found.")
    return model_names


def _resolve_datasets(dataset, datasets_config):
    if dataset:
        return [dataset]

    from datasets.loader import DatasetLoader

    dataset_names = DatasetLoader(datasets_config).dataset_names()
    if not dataset_names:
        raise click.ClickException(f"No datasets found in config: {datasets_config}")
    return dataset_names


@cli.command()
@click.argument("models", nargs=-1, required=False)
@click.option(
    "--dataset",
    required=False,
    help="Dataset to benchmark (configured name, HuggingFace ID, or local ImageFolder path). Default: all configured datasets.",
)
@click.option("--split", default="test", show_default=True)
@click.option("--task", default="classification", show_default=True, type=click.Choice(["classification"]))
@click.option("--device", default="cuda", show_default=True)
@click.pass_context
def run(ctx, models, dataset, split, task, device):
    """Run benchmark for one or more models on one or more datasets."""
    from benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        config_path=ctx.obj["config"],
        datasets_config_path=ctx.obj["datasets_config"],
    )
    model_names = _resolve_models(models)
    dataset_names = _resolve_datasets(dataset, ctx.obj["datasets_config"])

    for model_name in model_names:
        for dataset_name in dataset_names:
            runner.run(
                model_name=model_name,
                dataset=dataset_name,
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

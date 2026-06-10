# Vision Benchmark

A modular, extensible computer vision benchmarking framework with:

- **Plug-in model registry** — add a new model in one file with one decorator
- **Accumulated results** — every run appends to a SQLite DB + CSV; never overwritten
- **Error analysis** — per-class accuracy, high-confidence failures, confusion pairs, ECE calibration
- **W&B integration** — metrics, confusion matrix, failure images logged automatically

---

## Quick Start

```bash
pip install -r requirements.txt
# optional: install as CLI
pip install -e .
```

### Run a benchmark

```bash
# All registered models × all configured datasets (configs/datasets.yaml) on GPU
python benchmark.py run

# HuggingFace dataset
python benchmark.py run resnet50_timm --dataset cifar10 --split test

# Multiple models in one shot
python benchmark.py run resnet50_timm vit_base_timm --dataset cifar10

# Local ImageFolder dataset
python benchmark.py run resnet50_timm --dataset /path/to/my_dataset
```

### View the leaderboard

```bash
python benchmark.py leaderboard
```

### Re-run error analysis on a past run

```bash
python benchmark.py error-report <run_id>
```

### List registered models

```bash
python benchmark.py list-models
```

---

## W&B Setup

Set your project and entity in `configs/default.yaml`:

```yaml
wandb:
  enabled: true
  project: "vision-benchmark"
  entity: "your-wandb-username"
```

Or disable it:

```yaml
wandb:
  enabled: false
```

Log in once:

```bash
wandb login
```

---

## Adding a New Model

Create a file in `models/`, subclass `BaseVisionModel`, and register it:

```python
# models/my_model.py
from benchmarks.registry import ModelRegistry
from models.base import BaseVisionModel

@ModelRegistry.register("my_model")
class MyModel(BaseVisionModel):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        # load weights here

    def predict(self, images):
        # images: List[PIL.Image]
        # return: List[{"label": str, "conf": float, "probs": dict}]
        ...
```

That's it — no other changes needed. The runner auto-discovers all files in `models/`.

---

## Project Structure

```
Vision_benchmark/
├── benchmark.py          # CLI entrypoint
├── configs/
│   └── default.yaml      # All configuration
├── benchmarks/
│   ├── registry.py       # Model registry & auto-discovery
│   ├── runner.py         # Orchestrates a benchmark trial
│   ├── result_store.py   # SQLite persistence (accumulated results)
│   └── wandb_logger.py   # W&B integration
├── models/
│   ├── base.py           # BaseVisionModel interface
│   ├── resnet50_timm.py  # Example: ResNet-50 via timm
│   ├── vit_timm.py       # Example: ViT-Base via timm
│   └── clip_model.py     # Example: CLIP zero-shot
├── tasks/
│   └── classification.py # Classification task runner
├── error_analysis/
│   └── analyzer.py       # Error analysis & reporting
├── results/              # Auto-created: DB, CSVs, error reports
└── tests/                # pytest tests
```

---

## Results Accumulation

All runs persist to `results/results.db` (SQLite). Each run gets a unique `run_id`.
A `results/runs_summary.csv` is kept in sync after every run.

```python
from benchmarks.result_store import ResultStore
store = ResultStore()
print(store.leaderboard())
```

---

## Error Analysis Output

After every run, `results/error_analysis/<run_id>/` contains:

| File | Contents |
|------|----------|
| `report.json` | Full report: summary, per-class, failures, confusion pairs, ECE |
| `per_class.csv` | Per-class accuracy & confidence |
| `top_failures.csv` | Highest-confidence wrong predictions |
| `confusion_pairs.csv` | Most frequent misclassification pairs |

---

## Running Tests

```bash
pytest tests/ -v
```

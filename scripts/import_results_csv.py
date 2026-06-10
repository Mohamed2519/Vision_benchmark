#!/usr/bin/env python3
"""
Import previously-computed benchmark results from a flat CSV into the
benchmark's SQLite store, so the leaderboard / HTML report work without
re-running inference.

The source CSV holds one row per (model, dataset, split) where split is one
of {val, test, golden} with aggregate metrics and confusion-matrix counts:

    timestamp, model_name, dataset_name, split, n_samples_evaluated,
    accuracy, balanced_accuracy, sensitivity, specificity, precision, npv,
    f1, mcc, auroc, auprc, tp, tn, fp, fn, total, n_positive, n_negative

The pipeline, however, stores three split rows per (model, dataset):

    val_test   — val + test merged
    golden     — golden set
    combined   — everything pooled (drives the leaderboard)

Mapping strategy
----------------
* val_test  : sum the confusion counts of val + test, recompute all
              count-based metrics exactly; AUROC/AUPRC = sample-weighted
              average of the val & test values (raw scores are unavailable).
* golden    : copied directly.
* combined  : sum confusion counts across every available split, recompute
              count-based metrics; combined_auroc = sample-weighted average
              of all per-split AUROCs.

Run-id scheme matches the live pipeline so the HTML report finds the splits:
    <base>      -> combined row
    <base>_vt   -> val_test row
    <base>_g    -> golden row

Usage
-----
python scripts/import_results_csv.py \
    --csv results/imported_benchmark_results.csv \
    --db  results/results.db
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.result_store import ResultStore  # noqa: E402


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom > 0 else float("nan")


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """Exact count-based binary metrics (no scores needed)."""
    total = tp + tn + fp + fn
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity) \
        if (precision == precision and sensitivity == sensitivity) else float("nan")
    return {
        "accuracy": _safe_div(tp + tn, total),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "youden_j": sensitivity + specificity - 1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(total),
    }


def _weighted_avg(values: List[Optional[float]], weights: List[float]) -> Optional[float]:
    pairs = [(v, w) for v, w in zip(values, weights)
             if v is not None and v == v and w > 0]
    if not pairs:
        return None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    return num / den if den > 0 else None


@click.command()
@click.option("--csv", "csv_path", default="results/imported_benchmark_results.csv",
              show_default=True, help="Source CSV of precomputed results.")
@click.option("--db", "db_path", default="results/results.db",
              show_default=True, help="Target SQLite database.")
@click.option("--task", default="binary_classification", show_default=True)
def main(csv_path: str, db_path: str, task: str) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    store = ResultStore(db_path)

    n_pairs = 0
    for (model, dataset), grp in df.groupby(["model_name", "dataset_name"]):
        rows = {r["split"]: r for _, r in grp.iterrows()}

        base = str(uuid.uuid4())[:8]
        ts = str(grp["timestamp"].iloc[-1]) or datetime.utcnow().isoformat()

        # --- val_test = val + test merged ---
        vt_metrics = None
        vt_parts = [rows[s] for s in ("val", "test") if s in rows]
        if vt_parts:
            tp = sum(int(r["tp"]) for r in vt_parts)
            tn = sum(int(r["tn"]) for r in vt_parts)
            fp = sum(int(r["fp"]) for r in vt_parts)
            fn = sum(int(r["fn"]) for r in vt_parts)
            w = [int(r["total"]) for r in vt_parts]
            vt_metrics = _metrics_from_counts(tp, tn, fp, fn)
            vt_metrics["auroc"] = _weighted_avg([float(r["auroc"]) for r in vt_parts], w)
            vt_metrics["auprc"] = _weighted_avg([float(r["auprc"]) for r in vt_parts], w)

        # --- golden ---
        g_metrics = None
        if "golden" in rows:
            r = rows["golden"]
            g_metrics = _metrics_from_counts(int(r["tp"]), int(r["tn"]),
                                             int(r["fp"]), int(r["fn"]))
            g_metrics["auroc"] = float(r["auroc"])
            g_metrics["auprc"] = float(r["auprc"])

        # --- combined = all splits pooled ---
        all_parts = list(grp.itertuples(index=False))
        tp = sum(int(r.tp) for r in all_parts)
        tn = sum(int(r.tn) for r in all_parts)
        fp = sum(int(r.fp) for r in all_parts)
        fn = sum(int(r.fn) for r in all_parts)
        w_all = [int(r.total) for r in all_parts]
        combined_auroc = _weighted_avg([float(r.auroc) for r in all_parts], w_all)
        combined_auprc = _weighted_avg([float(r.auprc) for r in all_parts], w_all)

        combined_metrics = _metrics_from_counts(tp, tn, fp, fn)
        combined_metrics["auroc"] = combined_auroc
        combined_metrics["auprc"] = combined_auprc
        combined_metrics["combined_auroc"] = combined_auroc
        combined_metrics["val_test_auroc"] = vt_metrics["auroc"] if vt_metrics else None
        combined_metrics["golden_auroc"] = g_metrics["auroc"] if g_metrics else None
        combined_metrics["val_test_samples"] = vt_metrics["total"] if vt_metrics else 0
        combined_metrics["golden_samples"] = g_metrics["total"] if g_metrics else 0

        # --- persist (manually set timestamp to original) ---
        def save(run_id: str, split: str, metrics: Dict) -> None:
            metrics = {**metrics, "combined_auroc": combined_auroc}
            store._conn.execute(
                """INSERT INTO runs
                   (run_id, timestamp, model_name, task, dataset, split, metrics, config)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, ts, model, task, dataset, split,
                 json.dumps(metrics), json.dumps({"imported": True})),
            )

        if vt_metrics:
            save(base + "_vt", "val_test", vt_metrics)
        if g_metrics:
            save(base + "_g", "golden", g_metrics)
        save(base, "combined", combined_metrics)
        store._conn.commit()

        n_pairs += 1
        click.echo(f"  imported {model} / {dataset}  "
                   f"(combined_auroc={combined_auroc:.4f})")

    store.export_csv()
    click.echo(f"\nDone. Imported {n_pairs} model×dataset pairs into {db_path}")
    click.echo("Summary CSV → results/runs_summary.csv")


if __name__ == "__main__":
    main()

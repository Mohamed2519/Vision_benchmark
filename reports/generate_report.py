"""
HTML Benchmark Report Generator
================================

Produces a single self-contained HTML file with:

  1. Header — benchmark name, generation date, quick-stat cards
  2. Leaderboard table — all (model × dataset) combined results, sortable
  3. Combined AUROC bar chart — models ranked
  4. Per-dataset grouped bar — val_test vs golden AUROC per model
  5. Metrics heatmap — model × {AUROC, sensitivity, specificity, F1, ...}
  6. ROC-style scatter — sensitivity vs specificity per model/dataset bubble
  7. Confidence distribution — correct vs wrong violin/box per model
  8. Per-model detail cards — mini confusion matrices + per-split stats
  9. Top-50 failure gallery — worst high-confidence failures per run
 10. ECE calibration bars — per model

All charts are interactive (Plotly). No internet required after generation —
Plotly JS is inlined into the HTML.

Usage
-----
python reports/generate_report.py                          # reads results/results.db
python reports/generate_report.py --db path/to/results.db --out report.html
"""
from __future__ import annotations

import base64
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--db",  default="results/results.db", show_default=True,
              help="Path to the SQLite results database.")
@click.option("--out", default="results/benchmark_report.html", show_default=True,
              help="Output HTML file path.")
@click.option("--title", default="Chest X-Ray Benchmark Report", show_default=True)
def generate(db: str, out: str, title: str) -> None:
    """Generate a self-contained interactive HTML benchmark report."""
    db_path = Path(db)
    if not db_path.exists():
        click.echo(f"[error] Database not found: {db_path}", err=True)
        raise SystemExit(1)

    click.echo(f"Reading results from {db_path} …")
    builder = ReportBuilder(db_path=db_path, title=title)
    html = builder.build()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    click.echo(f"Report written → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    runs_df = pd.read_sql_query("SELECT * FROM runs", conn)
    preds_df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    if not runs_df.empty:
        runs_df["metrics"] = runs_df["metrics"].apply(json.loads)
        metrics_flat = pd.json_normalize(runs_df["metrics"])
        runs_df = pd.concat([runs_df.drop(columns=["metrics", "config"]), metrics_flat], axis=1)

    return runs_df, preds_df


def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _color_auc(v):
    """Return a CSS colour class for an AUROC value."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if v >= 0.90:
        return "cell-excellent"
    if v >= 0.80:
        return "cell-good"
    if v >= 0.70:
        return "cell-ok"
    return "cell-poor"


# ---------------------------------------------------------------------------
# Plotly helpers (returns JSON-serialisable dicts)
# ---------------------------------------------------------------------------

def _plotly_bar_auc(df_combined: pd.DataFrame) -> str:
    """Grouped bar: combined AUROC per model, one bar per dataset."""
    models   = sorted(df_combined["model_name"].unique())
    datasets = sorted(df_combined["dataset"].unique())

    PALETTE = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    ]

    traces = []
    for i, ds in enumerate(datasets):
        sub = df_combined[df_combined["dataset"] == ds]
        y   = [sub[sub["model_name"] == m]["combined_auroc"].values[0]
               if m in sub["model_name"].values else None for m in models]
        traces.append({
            "type": "bar",
            "name": ds,
            "x": models,
            "y": y,
            "marker": {"color": PALETTE[i % len(PALETTE)]},
            "text": [_fmt(v, 3) if v else "" for v in y],
            "textposition": "outside",
        })

    layout = {
        "title": {"text": "Combined AUROC by Model & Dataset", "font": {"size": 18}},
        "yaxis": {"title": "Combined AUROC", "range": [0, 1.05], "gridcolor": "#eee"},
        "xaxis": {"title": "Model"},
        "barmode": "group",
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "legend": {"title": {"text": "Dataset"}},
        "height": 420,
    }
    return json.dumps({"data": traces, "layout": layout})


def _plotly_split_auc(df_combined: pd.DataFrame) -> str:
    """Side-by-side val_test vs golden AUROC per model × dataset."""
    rows = df_combined[["model_name", "dataset", "val_test_auroc", "golden_auroc"]].copy()
    rows = rows.sort_values(["dataset", "model_name"])

    x_labels = [f"{r.model_name}<br><sub>{r.dataset}</sub>" for _, r in rows.iterrows()]

    traces = [
        {
            "type": "bar",
            "name": "val+test AUROC",
            "x": x_labels,
            "y": rows["val_test_auroc"].tolist(),
            "marker": {"color": "#4C72B0"},
        },
        {
            "type": "bar",
            "name": "golden AUROC",
            "x": x_labels,
            "y": rows["golden_auroc"].tolist(),
            "marker": {"color": "#DD8452"},
        },
    ]
    layout = {
        "title": {"text": "val+test vs Golden AUROC", "font": {"size": 18}},
        "yaxis": {"title": "AUROC", "range": [0, 1.05], "gridcolor": "#eee"},
        "barmode": "group",
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "height": 420,
    }
    return json.dumps({"data": traces, "layout": layout})


def _plotly_heatmap(df_combined: pd.DataFrame) -> str:
    """Heatmap: model+dataset rows × key metrics columns."""
    metric_cols = [
        "combined_auroc", "val_test_auroc", "golden_auroc",
        "auroc", "auprc", "accuracy", "sensitivity",
        "specificity", "f1", "youden_j",
    ]
    metric_cols = [c for c in metric_cols if c in df_combined.columns]

    df_combined = df_combined.copy()
    df_combined["row_label"] = df_combined["model_name"] + " / " + df_combined["dataset"]
    pivot = df_combined.set_index("row_label")[metric_cols].astype(float)

    # pandas >=2.1 renamed applymap → map; support both
    _applymap = getattr(pivot, "map", None) or pivot.applymap
    text = _applymap(lambda v: _fmt(v, 3)).values.tolist()

    data = [{
        "type": "heatmap",
        "z": pivot.values.tolist(),
        "x": metric_cols,
        "y": pivot.index.tolist(),
        "text": text,
        "texttemplate": "%{text}",
        "colorscale": "RdYlGn",
        "zmin": 0, "zmax": 1,
        "colorbar": {"title": "Score"},
    }]
    layout = {
        "title": {"text": "Metrics Heatmap", "font": {"size": 18}},
        "xaxis": {"title": "Metric", "tickangle": -30},
        "yaxis": {"title": "Model / Dataset", "automargin": True},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "height": max(350, 60 * len(pivot) + 120),
        "margin": {"l": 220},
    }
    return json.dumps({"data": data, "layout": layout})


def _plotly_sens_spec_bubble(df_combined: pd.DataFrame) -> str:
    """Scatter: sensitivity vs specificity, bubble = combined AUROC, colour = model."""
    models  = sorted(df_combined["model_name"].unique())
    PALETTE = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    ]
    traces = []
    for i, m in enumerate(models):
        sub = df_combined[df_combined["model_name"] == m]
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "name": m,
            "x": sub["specificity"].tolist(),
            "y": sub["sensitivity"].tolist(),
            "text": sub["dataset"].tolist(),
            "textposition": "top center",
            "marker": {
                "size": (sub["combined_auroc"].fillna(0) * 60).clip(10, 60).tolist(),
                "color": PALETTE[i % len(PALETTE)],
                "opacity": 0.8,
                "line": {"width": 1, "color": "white"},
            },
            "hovertemplate": (
                "<b>%{text}</b><br>"
                "Sensitivity: %{y:.3f}<br>"
                "Specificity: %{x:.3f}<extra>" + m + "</extra>"
            ),
        })

    # Ideal point
    traces.append({
        "type": "scatter", "mode": "markers",
        "name": "Ideal",
        "x": [1.0], "y": [1.0],
        "marker": {"symbol": "star", "size": 18, "color": "gold",
                   "line": {"width": 1, "color": "#333"}},
        "showlegend": True,
    })

    layout = {
        "title": {"text": "Sensitivity vs Specificity (bubble = Combined AUROC)", "font": {"size": 18}},
        "xaxis": {"title": "Specificity", "range": [0, 1.05], "gridcolor": "#eee"},
        "yaxis": {"title": "Sensitivity", "range": [0, 1.05], "gridcolor": "#eee"},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "height": 480,
    }
    return json.dumps({"data": traces, "layout": layout})


def _plotly_conf_dist(preds_df: pd.DataFrame, run_ids: List[str], model_name: str, dataset: str) -> str:
    """Box plot of confidence scores split by correct / wrong."""
    sub = preds_df[preds_df["run_id"].isin(run_ids)]
    if sub.empty:
        return json.dumps({"data": [], "layout": {"title": "No data"}})

    correct = sub[sub["correct"] == 1]["confidence"].dropna().tolist()
    wrong   = sub[sub["correct"] == 0]["confidence"].dropna().tolist()

    traces = [
        {
            "type": "violin", "name": "Correct",
            "y": correct, "box": {"visible": True},
            "line": {"color": "#55A868"}, "fillcolor": "#55A86866",
            "meanline": {"visible": True},
        },
        {
            "type": "violin", "name": "Wrong",
            "y": wrong, "box": {"visible": True},
            "line": {"color": "#C44E52"}, "fillcolor": "#C44E5266",
            "meanline": {"visible": True},
        },
    ]
    layout = {
        "title": {"text": f"Confidence Distribution — {model_name} / {dataset}", "font": {"size": 14}},
        "yaxis": {"title": "Confidence Score", "range": [0, 1]},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "height": 340,
        "margin": {"t": 50, "b": 40},
    }
    return json.dumps({"data": traces, "layout": layout})


def _plotly_ece_bar(df_combined: pd.DataFrame) -> str:
    """Bar chart of ECE per model × dataset (lower = better)."""
    if "ece" not in df_combined.columns:
        return json.dumps({"data": [], "layout": {"title": "ECE not available"}})

    sub = df_combined[["model_name", "dataset", "ece"]].dropna(subset=["ece"]).copy()
    sub["label"] = sub["model_name"] + " / " + sub["dataset"]
    sub = sub.sort_values("ece")

    data = [{
        "type": "bar",
        "x": sub["label"].tolist(),
        "y": sub["ece"].tolist(),
        "marker": {
            "color": sub["ece"].tolist(),
            "colorscale": "RdYlGn_r",
            "showscale": True,
            "colorbar": {"title": "ECE"},
        },
        "text": [_fmt(v, 4) for v in sub["ece"]],
        "textposition": "outside",
    }]
    layout = {
        "title": {"text": "Expected Calibration Error (lower = better)", "font": {"size": 18}},
        "yaxis": {"title": "ECE", "gridcolor": "#eee"},
        "xaxis": {"title": "Model / Dataset", "tickangle": -25},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
        "height": 400,
    }
    return json.dumps({"data": data, "layout": layout})


def _plotly_confusion(tp, tn, fp, fn) -> str:
    z      = [[tn, fp], [fn, tp]]
    text   = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    data = [{
        "type": "heatmap",
        "z": z,
        "x": ["Pred: Normal", "Pred: Abnormal"],
        "y": ["True: Normal", "True: Abnormal"],
        "text": text,
        "texttemplate": "%{text}",
        "colorscale": [[0, "#f7fbff"], [1, "#2166ac"]],
        "showscale": False,
    }]
    layout = {
        "height": 260,
        "margin": {"t": 30, "b": 40, "l": 110, "r": 20},
        "plot_bgcolor": "#fafafa",
        "paper_bgcolor": "#fff",
    }
    return json.dumps({"data": data, "layout": layout})


# ---------------------------------------------------------------------------
# Image thumbnail helper (base64 encode for inline HTML)
# ---------------------------------------------------------------------------

def _img_b64(path: str, max_px: int = 120) -> Optional[str]:
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(path).convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class ReportBuilder:
    def __init__(self, db_path: Path, title: str):
        self.title   = title
        self.runs_df, self.preds_df = _load_db(db_path)
        self._chart_idx = 0

    def _cid(self) -> str:
        self._chart_idx += 1
        return f"chart_{self._chart_idx}"

    def build(self) -> str:
        if self.runs_df.empty:
            return "<html><body><h1>No results in database yet.</h1></body></html>"

        df = self.runs_df
        combined = df[df["split"] == "combined"].copy()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # ---- Aggregate stats for header cards ----
        n_models   = combined["model_name"].nunique() if not combined.empty else 0
        n_datasets = combined["dataset"].nunique()    if not combined.empty else 0
        n_runs     = len(df[df["split"] == "combined"]) if not combined.empty else 0
        best_auroc = combined["combined_auroc"].max() if "combined_auroc" in combined.columns and not combined.empty else float("nan")
        best_row   = combined.loc[combined["combined_auroc"].idxmax()] if not combined.empty and "combined_auroc" in combined.columns else None

        sections = []

        # ---- Charts section ----
        if not combined.empty:
            cid1 = self._cid(); cid2 = self._cid(); cid3 = self._cid()
            cid4 = self._cid(); cid5 = self._cid()

            sections.append(f"""
<section class="section">
  <h2 class="section-title">AUROC Overview</h2>
  <div class="charts-row">
    <div class="chart-card half" id="{cid1}"></div>
    <div class="chart-card half" id="{cid2}"></div>
  </div>
  <div class="charts-row">
    <div class="chart-card full" id="{cid3}"></div>
  </div>
  <div class="charts-row">
    <div class="chart-card half" id="{cid4}"></div>
    <div class="chart-card half" id="{cid5}"></div>
  </div>
</section>
<script>
(function(){{
  var c1={_plotly_bar_auc(combined)};
  var c2={_plotly_split_auc(combined)};
  var c3={_plotly_heatmap(combined)};
  var c4={_plotly_sens_spec_bubble(combined)};
  var c5={_plotly_ece_bar(combined)};
  Plotly.newPlot('{cid1}', c1.data, c1.layout, {{responsive:true}});
  Plotly.newPlot('{cid2}', c2.data, c2.layout, {{responsive:true}});
  Plotly.newPlot('{cid3}', c3.data, c3.layout, {{responsive:true}});
  Plotly.newPlot('{cid4}', c4.data, c4.layout, {{responsive:true}});
  Plotly.newPlot('{cid5}', c5.data, c5.layout, {{responsive:true}});
}})();
</script>
""")

        # ---- Leaderboard table ----
        if not combined.empty:
            sections.append(self._leaderboard_section(combined))

        # ---- Per-model detail cards ----
        if not combined.empty:
            sections.append(self._model_detail_section(combined))

        # ---- Top-50 failure gallery ----
        sections.append(self._failure_gallery_section())

        # ---- Assemble HTML ----
        return HTML_TEMPLATE.format(
            title=self.title,
            generated=now,
            n_models=n_models,
            n_datasets=n_datasets,
            n_runs=n_runs,
            best_auroc=_fmt(best_auroc, 4),
            best_model=best_row["model_name"] if best_row is not None else "—",
            best_dataset=best_row["dataset"] if best_row is not None else "—",
            sections="\n".join(sections),
        )

    # ------------------------------------------------------------------
    def _leaderboard_section(self, combined: pd.DataFrame) -> str:
        cols_show = [
            ("model_name",     "Model"),
            ("dataset",        "Dataset"),
            ("combined_auroc", "Combined AUROC ↑"),
            ("val_test_auroc", "val+test AUROC"),
            ("golden_auroc",   "Golden AUROC"),
            ("auroc",          "AUROC"),
            ("auprc",          "AUPRC"),
            ("accuracy",       "Accuracy"),
            ("sensitivity",    "Sensitivity"),
            ("specificity",    "Specificity"),
            ("f1",             "F1"),
            ("youden_j",       "Youden J"),
            ("ece",            "ECE ↓"),
            ("total",          "Samples"),
            ("timestamp",      "Run date"),
        ]
        available = [(k, l) for k, l in cols_show if k in combined.columns]

        header = "".join(f"<th onclick=\"sortTable(this)\">{lbl} <span class='sort-icon'>⇅</span></th>" for _, lbl in available)

        sorted_df = combined.sort_values("combined_auroc", ascending=False) if "combined_auroc" in combined.columns else combined

        rows_html = ""
        for _, row in sorted_df.iterrows():
            cells = ""
            for col, _ in available:
                v = row.get(col)
                if col in ("model_name", "dataset", "timestamp", "total"):
                    cells += f"<td>{v if v is not None else '—'}</td>"
                else:
                    fv = _fmt(v, 4)
                    cls = _color_auc(v) if col in ("combined_auroc", "val_test_auroc", "golden_auroc", "auroc") else ""
                    cells += f"<td class='{cls}'>{fv}</td>"
            rows_html += f"<tr>{cells}</tr>"

        return f"""
<section class="section">
  <h2 class="section-title">Leaderboard</h2>
  <div class="table-wrap">
    <table class="results-table" id="leaderboard">
      <thead><tr>{header}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</section>
"""

    # ------------------------------------------------------------------
    def _model_detail_section(self, combined: pd.DataFrame) -> str:
        cards = []
        for _, row in combined.sort_values(["model_name", "dataset"]).iterrows():
            run_id = row["run_id"]
            model  = row["model_name"]
            ds     = row["dataset"]

            # per-split run_ids stored as run_id + _vt / _g
            run_ids = [run_id, run_id + "_vt", run_id + "_g"]

            cid_violin = self._cid()
            cid_cm     = self._cid()

            violin_json = _plotly_conf_dist(self.preds_df, run_ids, model, ds)

            tp = int(row.get("tp", 0) or 0)
            tn = int(row.get("tn", 0) or 0)
            fp = int(row.get("fp", 0) or 0)
            fn = int(row.get("fn", 0) or 0)
            cm_json = _plotly_confusion(tp, tn, fp, fn)

            stat_rows = ""
            for metric, label in [
                ("combined_auroc", "Combined AUROC"),
                ("val_test_auroc", "val+test AUROC"),
                ("golden_auroc",   "Golden AUROC"),
                ("sensitivity",    "Sensitivity"),
                ("specificity",    "Specificity"),
                ("f1",             "F1"),
                ("youden_j",       "Youden J"),
                ("ece",            "ECE"),
                ("total",          "Total samples"),
            ]:
                v = row.get(metric)
                fv = _fmt(v, 4)
                bar_pct = int(float(v) * 100) if isinstance(v, float) and not math.isnan(v) and 0 <= float(v) <= 1 else 0
                bar_html = f'<div class="mini-bar"><div class="mini-bar-fill" style="width:{bar_pct}%"></div></div>' if bar_pct else ""
                stat_rows += f"<tr><td class='stat-label'>{label}</td><td class='stat-val'>{fv}{bar_html}</td></tr>"

            cards.append(f"""
<div class="model-card">
  <div class="model-card-header">
    <span class="model-tag">{model}</span>
    <span class="dataset-tag">{ds}</span>
    <span class="auc-badge">AUROC {_fmt(row.get('combined_auroc'), 4)}</span>
  </div>
  <div class="model-card-body">
    <div class="model-stats">
      <table class="stat-table">{stat_rows}</table>
    </div>
    <div class="model-charts">
      <div id="{cid_cm}" class="mini-chart"></div>
      <div id="{cid_violin}" class="mini-chart"></div>
    </div>
  </div>
</div>
<script>
Plotly.newPlot('{cid_cm}',     {cm_json}.data,     {cm_json}.layout,     {{responsive:true,displayModeBar:false}});
Plotly.newPlot('{cid_violin}', {violin_json}.data, {violin_json}.layout, {{responsive:true,displayModeBar:false}});
</script>
""")

        return f"""
<section class="section">
  <h2 class="section-title">Per-Model Detail</h2>
  {"".join(cards)}
</section>
"""

    # ------------------------------------------------------------------
    def _failure_gallery_section(self) -> str:
        """Read top-50 failure CSVs from results/error_analysis/*/top_failures.csv."""
        base = Path("results/error_analysis")
        if not base.exists():
            return ""

        all_failures: List[Dict] = []
        for run_dir in sorted(base.iterdir()):
            csv_path = run_dir / "top_failures.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
                df["run_id"] = run_dir.name
                all_failures.append(df)
            except Exception:
                continue

        if not all_failures:
            return ""

        failures = pd.concat(all_failures, ignore_index=True)
        # sort by confidence descending across all runs
        if "confidence" in failures.columns:
            failures = failures.sort_values("confidence", ascending=False)

        # look up model/dataset from runs
        run_meta: Dict[str, tuple] = {}
        if not self.runs_df.empty:
            for _, r in self.runs_df.iterrows():
                run_meta[r["run_id"]] = (r["model_name"], r["dataset"])

        cards_html = ""
        for _, row in failures.iterrows():
            img_path = str(row.get("image_path", ""))
            label    = "abnormal" if str(row.get("label")) in ("1", "abnormal") else "normal"
            pred     = "abnormal" if str(row.get("pred"))  in ("1", "abnormal") else "normal"
            conf     = _fmt(row.get("confidence"), 3)
            rid      = str(row.get("run_id", ""))
            model_ds = f"{run_meta[rid][0]} / {run_meta[rid][1]}" if rid in run_meta else rid

            b64 = _img_b64(img_path) if img_path and Path(img_path).exists() else None
            img_html = (
                f'<img src="data:image/jpeg;base64,{b64}" class="failure-img" title="{img_path}">'
                if b64
                else f'<div class="failure-img no-img" title="{img_path}">No image</div>'
            )

            border_color = "#C44E52" if pred == "abnormal" else "#4C72B0"
            cards_html += f"""
<div class="failure-card" style="border-top: 3px solid {border_color}">
  {img_html}
  <div class="failure-meta">
    <span class="badge-true">True: {label}</span>
    <span class="badge-pred">Pred: {pred}</span>
    <div class="conf-score">conf {conf}</div>
    <div class="model-info">{model_ds}</div>
  </div>
</div>"""

        return f"""
<section class="section">
  <h2 class="section-title">Top Failure Gallery
    <span class="section-sub">Highest-confidence wrong predictions across all runs</span>
  </h2>
  <div class="failure-gallery">{cards_html}</div>
</section>
"""


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
/* ---- Reset & Base ---- */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f0f2f5;
  color: #1a1a2e;
  font-size: 14px;
}}
a {{ color: #4C72B0; text-decoration: none; }}

/* ---- Header ---- */
.report-header {{
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
  color: #fff;
  padding: 40px 48px 32px;
  position: relative;
  overflow: hidden;
}}
.report-header::after {{
  content: "";
  position: absolute;
  inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none;
}}
.header-title {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
.header-sub   {{ font-size: 13px; opacity: 0.6; margin-top: 4px; }}

/* ---- Stat Cards ---- */
.stat-cards {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 28px;
  position: relative;
  z-index: 1;
}}
.stat-card {{
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 12px;
  padding: 16px 22px;
  min-width: 140px;
  backdrop-filter: blur(4px);
}}
.stat-card .val {{ font-size: 26px; font-weight: 700; color: #e2b96f; }}
.stat-card .lbl {{ font-size: 11px; opacity: 0.7; margin-top: 2px; letter-spacing: 0.5px; text-transform: uppercase; }}
.stat-card.highlight .val {{ color: #6de8a4; font-size: 22px; }}

/* ---- Layout ---- */
.main-content {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}

/* ---- Sections ---- */
.section {{ margin-bottom: 48px; }}
.section-title {{
  font-size: 20px; font-weight: 700;
  border-left: 4px solid #4C72B0;
  padding-left: 12px;
  margin-bottom: 20px;
  display: flex; align-items: center; gap: 12px;
}}
.section-sub {{ font-size: 13px; font-weight: 400; color: #888; }}

/* ---- Charts ---- */
.charts-row {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
.chart-card {{
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 16px;
  overflow: hidden;
}}
.chart-card.half {{ flex: 1 1 calc(50% - 8px); min-width: 320px; }}
.chart-card.full {{ flex: 1 1 100%; }}

/* ---- Table ---- */
.table-wrap {{
  overflow-x: auto;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
}}
.results-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.results-table th {{
  background: #1a1a2e;
  color: #fff;
  padding: 11px 14px;
  text-align: left;
  cursor: pointer;
  white-space: nowrap;
  user-select: none;
}}
.results-table th:hover {{ background: #2a2a4e; }}
.results-table td {{
  padding: 9px 14px;
  border-bottom: 1px solid #f0f0f0;
  white-space: nowrap;
}}
.results-table tbody tr:hover {{ background: #f7f9ff; }}
.results-table tbody tr:nth-child(even) {{ background: #fafafa; }}
.results-table tbody tr:nth-child(even):hover {{ background: #f7f9ff; }}
.sort-icon {{ opacity: 0.5; font-size: 11px; }}
/* colour classes */
.cell-excellent {{ color: #1a7a4a; font-weight: 700; }}
.cell-good      {{ color: #2d7a2d; font-weight: 600; }}
.cell-ok        {{ color: #b36b00; }}
.cell-poor      {{ color: #c0392b; }}

/* ---- Model cards ---- */
.model-card {{
  background: #fff;
  border-radius: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,.07);
  margin-bottom: 20px;
  overflow: hidden;
}}
.model-card-header {{
  background: linear-gradient(90deg, #1a1a2e, #16213e);
  padding: 14px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}}
.model-tag   {{ background: #4C72B0; color:#fff; padding: 3px 10px; border-radius: 20px; font-size:13px; font-weight:600; }}
.dataset-tag {{ background: #DD8452; color:#fff; padding: 3px 10px; border-radius: 20px; font-size:12px; }}
.auc-badge   {{ margin-left: auto; background: #6de8a4; color:#1a1a2e; padding: 4px 14px; border-radius: 20px; font-size:13px; font-weight:700; }}
.model-card-body {{
  display: flex;
  gap: 0;
  flex-wrap: wrap;
}}
.model-stats {{ flex: 0 0 280px; padding: 16px 20px; border-right: 1px solid #f0f0f0; }}
.model-charts {{ flex: 1 1 0; display: flex; gap: 8px; flex-wrap: wrap; padding: 10px; }}
.mini-chart {{ flex: 1 1 calc(50% - 8px); min-width: 240px; }}

.stat-table {{ width: 100%; font-size: 13px; }}
.stat-table td {{ padding: 5px 0; }}
.stat-label {{ color: #666; width: 140px; }}
.stat-val   {{ font-weight: 600; }}
.mini-bar {{
  height: 4px;
  background: #eee;
  border-radius: 2px;
  margin-top: 3px;
}}
.mini-bar-fill {{
  height: 4px;
  background: linear-gradient(90deg, #4C72B0, #6de8a4);
  border-radius: 2px;
  transition: width .3s;
}}

/* ---- Failure gallery ---- */
.failure-gallery {{
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}}
.failure-card {{
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 10px;
  width: 140px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: box-shadow .2s;
}}
.failure-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.12); }}
.failure-img {{
  width: 120px; height: 120px;
  object-fit: cover;
  border-radius: 6px;
  background: #eee;
  display: block;
}}
.no-img {{
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: #aaa;
}}
.failure-meta {{ font-size: 11px; }}
.badge-true, .badge-pred {{
  display: inline-block;
  padding: 1px 6px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 600;
  margin-right: 2px;
}}
.badge-true {{ background: #e8f4e8; color: #2d7a2d; }}
.badge-pred {{ background: #fde8e8; color: #c0392b; }}
.conf-score {{ color: #888; margin-top: 3px; }}
.model-info {{ color: #aaa; font-size: 10px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

/* ---- Footer ---- */
.report-footer {{
  text-align: center;
  padding: 24px;
  color: #aaa;
  font-size: 12px;
  border-top: 1px solid #e8e8e8;
  margin-top: 32px;
}}

/* ---- Responsive ---- */
@media (max-width: 768px) {{
  .chart-card.half {{ flex: 1 1 100%; }}
  .model-stats {{ flex: 1 1 100%; border-right: none; border-bottom: 1px solid #f0f0f0; }}
}}
</style>
</head>
<body>

<!-- ====== HEADER ====== -->
<header class="report-header">
  <div class="header-title">{title}</div>
  <div class="header-sub">Generated {generated}</div>
  <div class="stat-cards">
    <div class="stat-card">
      <div class="val">{n_models}</div>
      <div class="lbl">Models evaluated</div>
    </div>
    <div class="stat-card">
      <div class="val">{n_datasets}</div>
      <div class="lbl">Datasets</div>
    </div>
    <div class="stat-card">
      <div class="val">{n_runs}</div>
      <div class="lbl">Total runs</div>
    </div>
    <div class="stat-card highlight">
      <div class="val">{best_auroc}</div>
      <div class="lbl">Best combined AUROC</div>
    </div>
    <div class="stat-card highlight">
      <div class="val" style="font-size:16px">{best_model}</div>
      <div class="lbl">Best model — {best_dataset}</div>
    </div>
  </div>
</header>

<!-- ====== MAIN ====== -->
<main class="main-content">
  {sections}
</main>

<!-- ====== FOOTER ====== -->
<footer class="report-footer">
  Vision Benchmark &nbsp;·&nbsp; Chest X-Ray Binary Classification &nbsp;·&nbsp; {generated}
</footer>

<!-- ====== TABLE SORT JS ====== -->
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const idx   = Array.from(th.parentNode.children).indexOf(th);
  const asc   = th.dataset.asc !== 'true';
  th.dataset.asc = asc;
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    const av = a.cells[idx].innerText.trim();
    const bv = b.cells[idx].innerText.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
  // update sort icons
  table.querySelectorAll('th .sort-icon').forEach(s => s.textContent = '⇅');
  th.querySelector('.sort-icon').textContent = asc ? '↑' : '↓';
}}
</script>

</body>
</html>
"""


if __name__ == "__main__":
    generate()

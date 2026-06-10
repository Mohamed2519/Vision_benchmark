"""
Error analysis module.

Surfaces:
- Per-class accuracy breakdown
- Confidence distribution for correct vs wrong predictions
- Top-N worst failures (lowest confidence correct / highest confidence wrong)
- Confusion pairs (most frequent misclassification pairs)
- Calibration: expected calibration error (ECE)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ErrorAnalyzer:
    def __init__(
        self,
        predictions: List[Dict],
        cfg: Dict[str, Any],
        run_id: str,
        output_dir: str = "results",
    ):
        self.preds = predictions
        self.cfg = cfg
        self.run_id = run_id
        self.output_dir = Path(output_dir) / "error_analysis" / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.DataFrame(predictions)

    def analyze(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}

        report["summary"] = self._summary()
        report["per_class"] = self._per_class_breakdown()
        report["top_failures"] = self._top_failures()
        report["confusion_pairs"] = self._confusion_pairs()
        report["ece"] = self._calibration_ece()
        report["confidence_stats"] = self._confidence_stats()

        # Persist
        with open(self.output_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        self._save_csvs(report)
        self._print_report(report)

        return report

    # ------------------------------------------------------------------
    def _summary(self) -> Dict:
        df = self.df
        total = len(df)
        correct = df["correct"].sum()
        return {
            "total": int(total),
            "correct": int(correct),
            "errors": int(total - correct),
            "accuracy": round(correct / total, 4) if total else 0.0,
            "avg_conf_correct": round(df[df["correct"] == 1]["confidence"].mean(), 4),
            "avg_conf_wrong": round(df[df["correct"] == 0]["confidence"].mean(), 4),
        }

    def _per_class_breakdown(self) -> List[Dict]:
        rows = []
        for cls, grp in self.df.groupby("label"):
            total = len(grp)
            correct = grp["correct"].sum()
            rows.append(
                {
                    "class": cls,
                    "total": int(total),
                    "correct": int(correct),
                    "accuracy": round(correct / total, 4),
                    "avg_conf": round(grp["confidence"].mean(), 4),
                    "avg_conf_wrong": round(
                        grp[grp["correct"] == 0]["confidence"].mean(), 4
                    ) if (grp["correct"] == 0).any() else None,
                }
            )
        return sorted(rows, key=lambda r: r["accuracy"])

    def _top_failures(self) -> List[Dict]:
        n = self.cfg.get("top_n_failures", 50)
        failures = self.df[self.df["correct"] == 0].copy()
        # Sort by confidence descending — high-confidence wrong = most egregious
        failures = failures.sort_values("confidence", ascending=False)
        cols = [c for c in ["sample_id", "image_path", "label", "pred", "confidence"] if c in failures.columns]
        return failures.head(n)[cols].to_dict("records")

    def _confusion_pairs(self, top_n: int = 20) -> List[Dict]:
        failures = self.df[self.df["correct"] == 0]
        counts = defaultdict(int)
        for _, row in failures.iterrows():
            counts[(row["label"], row["pred"])] += 1
        pairs = [
            {"true": k[0], "pred": k[1], "count": v}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        ]
        return pairs[:top_n]

    def _calibration_ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        confidences = self.df["confidence"].values
        correct = self.df["correct"].values
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidences)
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum() / total) * abs(acc - conf)
        return round(float(ece), 4)

    def _confidence_stats(self) -> Dict:
        df = self.df
        return {
            "mean": round(df["confidence"].mean(), 4),
            "std": round(df["confidence"].std(), 4),
            "p25": round(df["confidence"].quantile(0.25), 4),
            "p50": round(df["confidence"].quantile(0.50), 4),
            "p75": round(df["confidence"].quantile(0.75), 4),
        }

    def _save_csvs(self, report: Dict) -> None:
        pd.DataFrame(report["per_class"]).to_csv(
            self.output_dir / "per_class.csv", index=False
        )
        pd.DataFrame(report["top_failures"]).to_csv(
            self.output_dir / "top_failures.csv", index=False
        )
        pd.DataFrame(report["confusion_pairs"]).to_csv(
            self.output_dir / "confusion_pairs.csv", index=False
        )

    def _print_report(self, report: Dict) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        s = report["summary"]
        console.print(
            f"\n[bold yellow]Error Analysis[/] — "
            f"acc={s['accuracy']:.2%}  ECE={report['ece']}  "
            f"avg_conf_correct={s['avg_conf_correct']}  "
            f"avg_conf_wrong={s['avg_conf_wrong']}"
        )

        if report["confusion_pairs"]:
            table = Table(title="Top Confusion Pairs")
            table.add_column("True", style="green")
            table.add_column("Predicted", style="red")
            table.add_column("Count", style="cyan")
            for pair in report["confusion_pairs"][:10]:
                table.add_row(str(pair["true"]), str(pair["pred"]), str(pair["count"]))
            console.print(table)

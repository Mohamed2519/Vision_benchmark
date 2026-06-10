"""
Persistent result store backed by SQLite.

Every benchmark run appends rows — results accumulate across trials.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    task        TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    split       TEXT NOT NULL,
    metrics     TEXT NOT NULL,   -- JSON blob
    config      TEXT             -- JSON blob
);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    sample_id   TEXT,
    image_path  TEXT,
    label       TEXT,
    pred        TEXT,
    confidence  REAL,
    correct     INTEGER,
    extra       TEXT             -- JSON blob for task-specific fields
);
"""


class ResultStore:
    def __init__(self, db_path: str = "results/results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def save_run(
        self,
        run_id: str,
        model_name: str,
        task: str,
        dataset: str,
        split: str,
        metrics: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO runs (run_id, timestamp, model_name, task, dataset, split, metrics, config)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.utcnow().isoformat(),
                model_name,
                task,
                dataset,
                split,
                json.dumps(metrics),
                json.dumps(config or {}),
            ),
        )
        self._conn.commit()

    def save_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        rows = [
            (
                p["run_id"],
                p.get("sample_id"),
                p.get("image_path"),
                str(p.get("label")),
                str(p.get("pred")),
                float(p.get("confidence", 0.0)),
                int(p.get("correct", 0)),
                json.dumps(p.get("extra", {})),
            )
            for p in predictions
        ]
        self._conn.executemany(
            """INSERT INTO predictions
               (run_id, sample_id, image_path, label, pred, confidence, correct, extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get_runs(self, model_name: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM runs"
        params: tuple = ()
        if model_name:
            query += " WHERE model_name = ?"
            params = (model_name,)
        df = pd.read_sql_query(query, self._conn, params=params)
        if not df.empty:
            df["metrics"] = df["metrics"].apply(json.loads)
        return df

    def get_predictions(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM predictions WHERE run_id = ?",
            self._conn,
            params=(run_id,),
        )

    def export_csv(self, path: str = "results/runs_summary.csv") -> None:
        df = self.get_runs()
        # Flatten metrics dict into columns
        metrics_df = pd.json_normalize(df["metrics"])
        out = pd.concat([df.drop(columns=["metrics", "config"]), metrics_df], axis=1)
        out.to_csv(path, index=False)

    def leaderboard(self, task: str = "classification", metric: str = "top1_acc") -> pd.DataFrame:
        df = self.get_runs()
        df = df[df["task"] == task].copy()
        df[metric] = df["metrics"].apply(lambda m: m.get(metric))
        return (
            df[["model_name", "dataset", "split", metric, "timestamp"]]
            .sort_values(metric, ascending=False)
            .reset_index(drop=True)
        )

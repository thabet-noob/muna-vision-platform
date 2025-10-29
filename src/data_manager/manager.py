"""Data management utilities for writing prediction history.

This module provides a minimal CSV-based persistence layer used by services
to record aggregated predictions and metrics. The history directory is
configurable via the HISTORY_DIR environment variable to ease local runs
and container deployments.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Final
import os
import csv

_DEFAULT_HISTORY_DIR: Final[str] = os.environ.get("HISTORY_DIR", "/app/history")

# History directory and files are module-level so tests and services can
# monkeypatch them easily without changing call sites.
HISTORY: Path = Path(_DEFAULT_HISTORY_DIR)
HISTORY.mkdir(parents=True, exist_ok=True)
PRED_FILE: Path = HISTORY / "predictions.csv"

def append_prediction(people: int, students: int, happiness: float) -> None:
    """Append a single prediction row to the CSV history file.

    Parameters
    - people: Total number of people detected in the frame.
    - students: Number of detected people classified as students.
    - happiness: Aggregated happiness score in [0, 100].

    Notes
    - The file is created on first write with a header.
    - "non_students" is computed as max(0, people - students).
    """
    new_file = not PRED_FILE.exists()
    with open(PRED_FILE, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if new_file:
            writer.writerow(["ts", "people", "students", "non_students", "happiness"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            int(people),
            int(students),
            max(0, int(people) - int(students)),
            round(float(happiness), 2),
        ])

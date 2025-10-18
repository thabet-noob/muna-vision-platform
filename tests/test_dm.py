from src.data_manager.manager import append_prediction
from pathlib import Path
import csv

def test_append_prediction(tmp_path, monkeypatch):
    # Patch history path
    from src import data_manager
    monkeypatch.setattr(data_manager.manager, "HISTORY", tmp_path)
    monkeypatch.setattr(data_manager.manager, "PRED_FILE", tmp_path / "pred.csv")
    append_prediction(3,2,77.7)
    assert (tmp_path / "pred.csv").exists()
    with open(tmp_path / "pred.csv") as f:
        lines = list(csv.reader(f))
        assert len(lines) == 2  # header + one row

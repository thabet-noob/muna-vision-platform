from pathlib import Path
from datetime import datetime
import csv

HISTORY = Path("/app/history")
HISTORY.mkdir(parents=True, exist_ok=True)
PRED_FILE = HISTORY / "predictions.csv"

def append_prediction(people: int, students: int, happiness: float):
    new = not PRED_FILE.exists()
    with open(PRED_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts","people","students","non_students","happiness"])
        w.writerow([datetime.utcnow().isoformat(), people, students, max(0, people-students), round(float(happiness),2)])

from prefect import flow, task
from datetime import datetime
from pathlib import Path
import csv
import random

HISTORY = Path("/app/history")
HISTORY.mkdir(parents=True, exist_ok=True)
PRED_FILE = HISTORY / "predictions.csv"

@task
def simulate_prediction():
    # Fake periodic metric when pipeline isn't connected
    people = random.randint(0, 12)
    students = random.randint(0, people)
    happy = round(random.uniform(35, 90), 2)
    return people, students, happy

@task
def append_history(people, students, happiness):
    new = not PRED_FILE.exists()
    with open(PRED_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts","people","students","non_students","happiness"])
        w.writerow([datetime.utcnow().isoformat(), people, students, people-students, happiness])

@flow(name="inference_flow")
def inference_flow():
    p, s, h = simulate_prediction()
    append_history(p, s, h)
    print(f"[inference_flow] p={p} s={s} h={h}")

if __name__ == "__main__":
    inference_flow()

from prefect import flow, task
from pathlib import Path
import pandas as pd
import requests

HISTORY = Path("/app/history")
PRED_FILE = HISTORY / "predictions.csv"

@task
def compute_daily_metrics():
    if not PRED_FILE.exists():
        return None
    df = pd.read_csv(PRED_FILE)
    if df.empty:
        return None
    return {
        "count": int(len(df)),
        "people_avg": float(df["people"].mean()),
        "students_avg": float(df["students"].mean()),
        "happiness_avg": float(df["happiness"].mean()),
    }

@task
def post_to_monitoring(metrics: dict):
    if not metrics: 
        return
    try:
        requests.post("http://monitoring_api:8002/ingest", json=metrics, timeout=3)
    except Exception as e:
        print("post_to_monitoring error:", e)

@flow(name="monitor_flow")
def monitor_flow():
    m = compute_daily_metrics()
    post_to_monitoring(m)
    print(f"[monitor_flow] {m}")

if __name__ == "__main__":
    monitor_flow()

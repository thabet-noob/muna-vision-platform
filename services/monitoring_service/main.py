from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import datetime as dt
import os

app = FastAPI(title="Monitoring API")
HISTORY_DIR = os.environ.get("HISTORY_DIR", "/app/history")
HISTORY = Path(HISTORY_DIR)
PRED_FILE = HISTORY / "predictions.csv"
METRICS_FILE = HISTORY / "metrics.csv"
HISTORY.mkdir(parents=True, exist_ok=True)

class Metrics(BaseModel):
    count: int
    people_avg: float
    students_avg: float
    happiness_avg: float

@app.get("/health")
def health() -> dict:
    return {"status":"ok"}

@app.post("/ingest")
def ingest(m: Metrics):
    new = not METRICS_FILE.exists()
    df = pd.DataFrame([{
        "ts": dt.datetime.utcnow().isoformat(),
        "count": m.count,
        "people_avg": m.people_avg,
        "students_avg": m.students_avg,
        "happiness_avg": m.happiness_avg,
    }])
    if new:
        df.to_csv(METRICS_FILE, index=False)
    else:
        df.to_csv(METRICS_FILE, mode="a", header=False, index=False)
    return {"ok": True}

@app.get("/metrics")
def metrics():
    if not METRICS_FILE.exists():
        return []
    return pd.read_csv(METRICS_FILE).to_dict(orient="records")

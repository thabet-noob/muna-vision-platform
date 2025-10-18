from prefect import flow, task
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODELS = Path("/app/models")
MODELS.mkdir(parents=True, exist_ok=True)

@task
def generate_synthetic_data(n=500):
    # x: yellow ratio in [0,1], y: 1 if student
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(n,1))
    y = (x[:,0] + 0.15*rng.standard_normal(n) > 0.28).astype(int)
    return x, y

@task
def train_model(x, y):
    clf = LogisticRegression()
    clf.fit(x, y)
    return clf

@task
def save_model(clf):
    out = MODELS / "student_calib.pkl"
    joblib.dump(clf, out)
    return str(out)

@flow(name="train_flow")
def train_flow():
    x, y = generate_synthetic_data()
    clf = train_model(x, y)
    path = save_model(clf)
    print(f"[train_flow] saved: {path}")

if __name__ == "__main__":
    train_flow()

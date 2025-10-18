from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_demo(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=(800,1))
    y = (x[:,0] + 0.12*rng.standard_normal(800) > 0.28).astype(int)
    clf = LogisticRegression()
    clf.fit(x, y)
    out = output_dir / "student_calib.pkl"
    joblib.dump(clf, out)
    return str(out)

from src.pipelines.training.train_pipeline import train_demo
from pathlib import Path

def main():
    out = train_demo(Path("/app/models"))
    print("[training_service] saved", out)

if __name__ == "__main__":
    main()

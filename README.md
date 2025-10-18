# Muna Vision Platform — Full ML Project (Training + Inference + Dashboard + Prefect)

This repository packages your **Muna School Academy** computer-vision dashboard into a production-style ML project,
inspired by a typical platform blueprint. It includes:

- **Pipelines**: preprocessing, feature eng, inference, postprocessing (decoupled from the app).
- **Services**: Dashboard (Dash), Inference API (FastAPI), Training job (Sklearn demo), Monitoring API (FastAPI).
- **Prefect Orchestrator**: training + monitoring flows and simple deployments for local runs.
- **Volumes**: data, models, predictions history (shared via Docker volumes).
- **CI**: minimal GitHub Actions workflow for lint/tests.
- **Docker**: per-service Dockerfiles + `docker-compose.yml` to run everything locally.

The structure matches the high-level architecture in your diagrams (Git repo → pipelines, data/model monitoring, CI/CD; prod
env with **training**/**inference** containers, a **dashboard** pulling data and history, and **shared volumes** for data/models).

---

## Quickstart (Docker)

1. Make sure Docker is running.
2. Clone/copy this repo and go inside the folder.
3. (Optional) Put your YOLO weights in `volumes/model_store/yolov8n.pt` or let Ultralytics download them automatically.
4. Set your camera URL in `.env` (copy from `.env.example`).

```bash
cp .env.example .env
# edit .env to set RTSP_URL
docker compose up --build
```

This will start:
- `dashboard` on **http://localhost:8050**
- `inference_api` on **http://localhost:8001**
- `monitoring_api` on **http://localhost:8002**
- `orchestrator` running Prefect flows on a schedule (local mode)
- `trainer` runs once to (re)train a simple calibrator model

> Notes
> - First run will download PyTorch/Ultralytics weights (internet required). If you prefer offline, place the file
>   `yolov8n.pt` under `volumes/model_store/` and export `YOLO_WEIGHTS=/app/models/yolov8n.pt` in `.env`.
> - **CPU build** of PyTorch is used. Modify the base image or requirements if you have CUDA available.

---

## Repository Layout

```
muna-vision-platform/
├─ .github/workflows/ci.yml               # lint + tests
├─ orchestration/
│  ├─ flows/
│  │  ├─ train_flow.py
│  │  ├─ inference_flow.py
│  │  └─ monitor_flow.py
│  ├─ start_local.py                      # orchestration entrypoint used by the container
│  └─ README.md
├─ services/
│  ├─ dashboard/
│  │  ├─ Dockerfile
│  │  ├─ requirements.txt
│  │  ├─ app.py                           # Dash UI (your provided app, refactored to call pipeline funcs)
│  │  └─ assets/muna.jpg                  # placeholder avatar
│  ├─ inference_service/
│  │  ├─ Dockerfile
│  │  ├─ requirements.txt
│  │  └─ main.py                          # FastAPI: /predict and /health
│  ├─ training_service/
│  │  ├─ Dockerfile
│  │  ├─ requirements.txt
│  │  └─ train.py                         # simple demo model training (sklearn LogisticRegression)
│  └─ monitoring_service/
│     ├─ Dockerfile
│     ├─ requirements.txt
│     └─ main.py                          # FastAPI: ingest + view metrics (CSV-backed)
├─ src/
│  ├─ common/video.py                     # RTSP video stream
│  ├─ data_manager/manager.py             # append predictions to CSV
│  └─ pipelines/
│     ├─ inference/
│     │  ├─ color.py                      # yellow ratio + student rule
│     │  ├─ happiness.py                  # MediaPipe FaceMesh + OpenCV fallback
│     │  ├─ yolo.py                       # ultralytics loader + predict
│     │  └─ pipeline.py                   # glue used by app + inference service
│     └─ training/
│        └─ train_pipeline.py             # demo training API
├─ tests/
│  ├─ test_color.py
│  └─ test_dm.py
├─ docker-compose.yml
├─ .env.example
├─ LICENSE
├─ README.md
└─ .gitignore
```

---

## Environment variables

Create `.env` from `.env.example` and adjust:

- `RTSP_URL` — your camera RTSP (e.g., `rtsp://USER:PASS@192.168.1.207:554/live1`).
- `YOLO_WEIGHTS` — optional path to yolov8 weights (inside container), default uses Ultralytics CDN.
- `TICK_MS` — dashboard refresh (ms), default `500`.
- `IMGSZ` — YOLO image size, default `640`.

---

## How it Works (Design)

- **Pipelines in `src/pipelines`**:
  - `inference/` is split into stages: `color.py` (yellow rule), `yolo.py` (person detection),
    `happiness.py` (FaceMesh or cascades), and `pipeline.py` (composes them).
  - `training/` provides a **demo** supervised stage that "calibrates" the yellow-ratio rule using logistic regression
    (purely illustrative; your main model is pre‑trained YOLO). The trained sklearn model is stored under
    `volumes/model_store/student_calib.pkl`.

- **Services**:
  - `dashboard/` embeds your Dash app and imports the pipeline functions instead of duplicating code.
  - `inference_service/` exposes a REST API to run the inference pipeline from other clients (e.g., batch jobs, tests).
  - `training_service/` runs the demo training once on container start (or trigger via Prefect).
  - `monitoring_service/` collects and serves simple metrics computed from `volumes/history/predictions.csv`.

- **Prefect orchestrator** (`orchestration/`):
  - `train_flow.py`: runs the training pipeline on a schedule and saves artifacts under `model_store`.
  - `inference_flow.py`: samples a frame from RTSP or local image and logs predictions (simulated batch score).
  - `monitor_flow.py`: aggregates daily statistics (counts, mean happiness) and posts to `monitoring_service`.
  - `start_local.py`: a lightweight runner that starts flows periodically in-process (no external Prefect server required).
    If you prefer the full Prefect UI, swap to `prefect server start` and build deployments.

- **Volumes** (Docker): `./volumes/model_store`, `./volumes/history`, `./volumes/inference_data` are shared across services
  to emulate **model registry**, **predictions history**, and **data landing** areas shown in the diagrams.

---

## Local (non-Docker) runs

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r services/dashboard/requirements.txt
export RTSP_URL="rtsp://USER:PASS@CAM_IP/live1"
python services/dashboard/app.py
```

For training:

```bash
pip install -r services/training_service/requirements.txt
python services/training_service/train.py
```

For the inference API:

```bash
pip install -r services/inference_service/requirements.txt
uvicorn services.inference_service.main:app --reload --port 8001
```

---

## CI

GitHub Actions in `.github/workflows/ci.yml` run `flake8` and `pytest` on push/PR.

---

## Security & Next steps

- Consider moving to a proper **model registry** (MLflow, S3/MinIO) instead of filesystem volumes.
- Replace the demo calibrator with a real **student uniform classifier** or face-based re-id if needed.
- Hardening: RTSP credentials via Docker secrets, add auth to APIs, central logging/metrics (Prometheus).

---

© 2025

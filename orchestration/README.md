This directory contains Prefect 2.x flows. For simplicity in Docker, we run them directly via `start_local.py`.
If you want the full Prefect UI:

1) `pip install prefect`
2) `prefect server start`  # in a separate terminal
3) Build deployments from the flows:
   - `prefect deployment build orchestration/flows/train_flow.py:train_flow -n train_local -q default -o train-deploy.yaml`
   - `prefect deployment apply train-deploy.yaml`
4) Start a worker:
   - `prefect worker start -q default`

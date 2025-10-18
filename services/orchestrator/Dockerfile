FROM python:3.10-slim
WORKDIR /app
COPY orchestration /app/orchestration
COPY services/orchestrator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
CMD ["python","/app/orchestration/start_local.py"]

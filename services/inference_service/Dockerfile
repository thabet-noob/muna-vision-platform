FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app

COPY services/inference_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
EXPOSE 8001
CMD ["uvicorn","services.inference_service.main:app","--host","0.0.0.0","--port","8001"]

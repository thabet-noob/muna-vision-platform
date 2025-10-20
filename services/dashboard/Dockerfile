FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
WORKDIR /app

COPY services/dashboard/requirements.txt /app/requirements.txt
# nuke any GUI wheel from cache, then install headless
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
EXPOSE 8050
CMD ["python","services/dashboard/app.py"]

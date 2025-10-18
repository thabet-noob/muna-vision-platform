FROM python:3.10-slim
WORKDIR /app
COPY services/training_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
CMD ["python","services/training_service/train.py"]

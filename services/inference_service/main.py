from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, cv2, io, os
from src.pipelines.inference.pipeline import detect_and_annotate, compute_happiness, to_jpeg

app = FastAPI(title="Inference API")

@app.get("/health")
def health():
    return {"status":"ok"}

class PredictResponse(BaseModel):
    people: int
    students: int
    non_students: int
    happiness: float
    annotated_b64: str

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    annotated, people, students, head_rois = detect_and_annotate(frame)
    happiness = float(compute_happiness(frame, head_rois))
    return {
        "people": int(people),
        "students": int(students),
        "non_students": int(max(0, people-students)),
        "happiness": float(happiness),
        "annotated_b64": to_jpeg(annotated),
    }

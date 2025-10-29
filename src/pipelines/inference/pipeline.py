"""End-to-end inference helpers for detection, annotation and encoding."""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2
from .yolo import load_yolo
from .color import student_by_yellow
from .happiness import compute_happiness

def detect_and_annotate(
    frame: np.ndarray,
    imgsz: int = 640,
    conf: float = 0.35,
    device_hint: Optional[str] = None,
):
    """Detect persons, annotate the frame, and collect head ROIs.

    Returns (annotated_frame, people_count, students_count, head_rois).
    """
    model, device = load_yolo(device_hint=device_hint)
    res = model.predict(frame, imgsz=imgsz, conf=conf, iou=0.5, classes=[0], device=device, verbose=False)[0]
    annotated = res.plot()
    people = 0
    students = 0
    head_rois: List[np.ndarray] = []

    H, W = frame.shape[:2]
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            roi = frame[y1:y2, x1:x2]
            people += 1
            is_student, yratio = student_by_yellow(roi)
            if is_student:
                students += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
                tag = f"student {int(yratio*100)}% yellow"
                cv2.putText(annotated, tag, (x1, max(0, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            pw, ph = (x2 - x1), (y2 - y1)
            hx1 = x1 + int(pw * 0.28)
            hx2 = x1 + int(pw * 0.72)
            hy2 = y1 + int(ph * 0.50)
            head = frame[y1:hy2, hx1:hx2]
            if head.size > 0:
                head_rois.append(head)
    return annotated, people, students, head_rois

def to_jpeg(img: np.ndarray, size: Tuple[int, int] = (960, 540)) -> Optional[str]:
    """Encode an image as a base64 JPEG data URL; returns None on failure."""
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(enc).decode("utf-8")

__all__ = ["detect_and_annotate", "compute_happiness", "to_jpeg"]

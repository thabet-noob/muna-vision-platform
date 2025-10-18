import cv2
import numpy as np
import mediapipe as mp

_mp_mesh = None
_face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def _ensure_facemesh():
    global _mp_mesh
    if _mp_mesh is None:
        _mp_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def _happy_from_mesh(landmarks) -> float:
    L = landmarks.landmark
    xL, yL = L[61].x, L[61].y
    xR, yR = L[291].x, L[291].y
    xU, yU = L[13].x, L[13].y
    xD, yD = L[14].x, L[14].y
    mouth_w = np.hypot(xR - xL, yR - yL) + 1e-6
    mouth_h = np.hypot(xD - xU, yD - yU)
    y_center = (yU + yD) * 0.5
    y_corners = (yL + yR) * 0.5
    curvature = _clamp((y_center - y_corners) / 0.04)
    open_ratio = mouth_h / mouth_w
    open_norm = _clamp((open_ratio - 0.08) / 0.22)
    score = 50.0 + 50.0 * _clamp(0.8 * curvature + 0.2 * open_norm)
    return float(max(0.0, min(100.0, score)))

_happy_score = 50.0

def compute_happiness(frame, head_rois=None) -> float:
    global _happy_score
    try:
        _ensure_facemesh()
        h, w = frame.shape[:2]
        scale = 800.0 / max(1.0, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = _mp_mesh.process(rgb)
        scores = []
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks[:8]:
                scores.append(_happy_from_mesh(lm))
        if scores:
            avg = float(np.mean(scores))
            _happy_score = 0.7 * _happy_score + 0.3 * avg
            return float(_happy_score)
    except Exception as e:
        print("MediaPipe error:", e)
    scores = []
    rois = list(head_rois or [])
    if not rois:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_full = _face_cascade.detectMultiScale(gray_full, scaleFactor=1.08, minNeighbors=6, minSize=(60, 60))
        for (x, y, w, h) in faces_full[:8]:
            rois.append(frame[y:y+h, x:x+w])
    for roi in rois[:8]:
        if roi is None or roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        smiles = _smile_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=10, minSize=(22, 22))
        scores.append(100.0 if len(smiles) > 0 else 50.0)
    if scores:
        avg = float(np.mean(scores))
        _happy_score = 0.7 * _happy_score + 0.3 * avg
    return float(_happy_score)

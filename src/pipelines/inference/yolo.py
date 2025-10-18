from ultralytics import YOLO
import torch

_device = "cpu"
_model = None

def device():
    global _device
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = "mps"
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
    except Exception:
        _device = "cpu"
    return _device

def load_yolo(weights: str | None = None):
    global _model
    if _model is None:
        _ = device()
        _model = YOLO(weights or "yolov8n.pt")
    return _model, _device

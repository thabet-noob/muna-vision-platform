"""YOLO model loading utilities with lightweight device selection."""

from __future__ import annotations

from typing import Optional, Tuple
from ultralytics import YOLO
import torch

_device: str = "cpu"
_model = None

def device() -> str:
    """Detect the best-available device among mps, cuda, cpu."""
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

def _validate_device_hint(hint: Optional[str]) -> Optional[str]:
    if hint is None:
        return None
    normalized = hint.lower().strip()
    if normalized not in {"cpu", "cuda", "mps"}:
        return None
    if normalized == "cuda" and not torch.cuda.is_available():
        return None
    if normalized == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return None
    return normalized


def load_yolo(weights: Optional[str] = None, device_hint: Optional[str] = None) -> Tuple[YOLO, str]:
    """Load a singleton YOLO model and return it with the selected device.

    Parameters
    - weights: Optional custom weights path; defaults to "yolov8n.pt".
    - device_hint: Optional device preference among {"cpu", "cuda", "mps"}.
      If not available, falls back to automatic detection.
    """
    global _model, _device
    hint = _validate_device_hint(device_hint)
    if hint is not None:
        _device = hint
    if _model is None:
        # Ensure _device is initialized if no valid hint was provided
        if hint is None:
            _ = device()
        _model = YOLO(weights or "yolov8n.pt")
    return _model, _device

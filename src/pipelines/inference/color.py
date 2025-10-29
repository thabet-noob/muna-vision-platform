"""Color-based utilities to identify student uniform presence.

Current heuristic looks for a sufficiently large yellow area in shirt/skirt
regions of a person ROI. All functions are defensive against empty inputs.
"""

from __future__ import annotations

from typing import Final, Tuple
import numpy as np
import cv2

Y_LOWER: Final[np.ndarray] = np.array([15, 70, 70], dtype=np.uint8)
Y_UPPER: Final[np.ndarray] = np.array([40, 255, 255], dtype=np.uint8)
Y_RATIO_THR: Final[float] = 0.12

def yellow_ratio(bgr: np.ndarray) -> float:
    """Compute the fraction of yellow pixels in a BGR image.

    Returns 0.0 for empty inputs.
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Y_LOWER, Y_UPPER)
    return float(np.count_nonzero(mask)) / float(mask.size)

def student_by_yellow(roi: np.ndarray) -> Tuple[bool, float]:
    """Heuristically classify a person ROI as a student based on yellow coverage.

    Looks at two sub-regions (shirt and skirt) and uses the maximum yellow ratio
    as the score. Returns a boolean decision and the ratio used.
    """
    if roi is None or roi.size == 0:
        return False, 0.0
    h, w = roi.shape[:2]
    if h < 20 or w < 20:
        return False, 0.0
    shirt = roi[int(h * 0.20) : int(h * 0.65), int(w * 0.20) : int(w * 0.80)]
    skirt = roi[int(h * 0.55) : h,             int(w * 0.20) : int(w * 0.80)]
    r1 = yellow_ratio(shirt)
    r2 = yellow_ratio(skirt)
    r = max(r1, r2)
    return (r >= Y_RATIO_THR), r

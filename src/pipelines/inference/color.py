import numpy as np
import cv2

Y_LOWER = np.array([15, 70, 70], dtype=np.uint8)
Y_UPPER = np.array([40, 255, 255], dtype=np.uint8)
Y_RATIO_THR = 0.12

def yellow_ratio(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Y_LOWER, Y_UPPER)
    return float(np.count_nonzero(mask)) / float(mask.size)

def student_by_yellow(roi: np.ndarray):
    if roi is None or roi.size == 0:
        return False, 0.0
    h, w = roi.shape[:2]
    if h < 20 or w < 20:
        return False, 0.0
    shirt = roi[int(h*0.20):int(h*0.65), int(w*0.20):int(w*0.80)]
    skirt = roi[int(h*0.55):h,           int(w*0.20):int(w*0.80)]
    r1 = yellow_ratio(shirt)
    r2 = yellow_ratio(skirt)
    r = max(r1, r2)
    return (r >= Y_RATIO_THR), r

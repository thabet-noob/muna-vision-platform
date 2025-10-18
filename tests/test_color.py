import numpy as np
from src.pipelines.inference.color import yellow_ratio
import cv2

def test_yellow_ratio_basic():
    # pure yellow in HSV approx (30, 255, 255)
    img_hsv = np.full((10,10,3), (30,255,255), dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    r = yellow_ratio(img_bgr)
    assert r > 0.9

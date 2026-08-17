import cv2
import numpy as np
from typing import Optional
from .types import CropConfig
from .contour import largest_valid_contour, contour_to_quad

def classic_contour_detect(image_bgr: np.ndarray, cfg: CropConfig) -> Optional[np.ndarray]:
    """
    Pipeline cổ điển: Canny edge detection + findContours + approxPolyDP.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Khử nhiễu nhẹ để Canny hoạt động tốt hơn
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny với Otsu threshold estimation
    v = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    # Giãn nở nhẹ để liền viền
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    c = largest_valid_contour(edged, cfg)
    if c is not None:
        return contour_to_quad(c, cfg)
        
    return None

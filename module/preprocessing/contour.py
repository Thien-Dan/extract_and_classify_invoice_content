import cv2
import numpy as np
from typing import Optional
from .types import CropConfig
from .geometry import is_valid_quad

def clean_mask(mask: np.ndarray, cfg: CropConfig) -> np.ndarray:
    """
    Threshold mask (Otsu mặc định) và Morphology (closing rồi opening).
    """
    if cfg.mask_threshold is None:
        _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(mask, cfg.mask_threshold, 255, cv2.THRESH_BINARY)

    # Morphology: closing rồi opening để nối vùng đứt / loại nhiễu nhỏ.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_kernel_size, cfg.morph_kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    return opened

def largest_valid_contour(binary_mask: np.ndarray, cfg: CropConfig):
    """
    findContours → lọc theo % diện tích ảnh (min/max ratio có thể cấu hình).
    Lưu ý: hóa đơn chụp cận cảnh có thể chiếm >99% khung hình.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = binary_mask.shape[0] * binary_mask.shape[1]
    
    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / float(img_area)
        if cfg.min_area_ratio <= ratio <= cfg.max_area_ratio:
            return c
            
    return None

def contour_to_quad(contour, cfg: CropConfig) -> Optional[np.ndarray]:
    """
    approxPolyDP ép contour lớn nhất về tứ giác (4 điểm).
    Nếu không ra đúng 4 điểm, fallback sang minAreaRect trên convex hull của contour.
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, cfg.approx_epsilon_ratio * peri, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
        
    # Fallback: minAreaRect trên convex hull
    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    return box.astype(np.float32)

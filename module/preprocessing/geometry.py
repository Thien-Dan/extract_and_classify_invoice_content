import numpy as np
import cv2
from typing import Optional
from .types import CropConfig

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def expand_quad(quad: np.ndarray, image_shape: tuple, expand_ratio: float) -> np.ndarray:
    """
    Mở rộng tứ giác ra ngoài một chút (theo tỉ lệ expand_ratio) từ tâm của tứ giác.
    Giúp tránh cắt phạm vào chữ sát lề.
    """
    if expand_ratio <= 0:
        return quad
        
    h, w = image_shape[:2]
    center = np.mean(quad, axis=0)
    
    expanded = center + (quad - center) * (1.0 + expand_ratio)
    
    # Clip lại vào trong giới hạn của ảnh gốc
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    
    return expanded.astype(np.float32)

def is_valid_quad(quad: np.ndarray, image_shape: tuple, cfg: CropConfig) -> bool:
    """
    Kiểm tra sanity: tỉ lệ cạnh (aspect ratio) của tứ giác phải nằm trong range hợp lý.
    """
    rect = order_points(quad)
    (tl, tr, br, bl) = rect

    # Tính width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Tính height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth == 0 or maxHeight == 0:
        return False

    aspect_ratio = maxHeight / float(maxWidth)
    return cfg.min_aspect_ratio <= aspect_ratio <= cfg.max_aspect_ratio

def four_point_transform(image: np.ndarray, pts: np.ndarray, upscale_factor: float = 1.0) -> np.ndarray:
    """
    Four-point perspective transform từ ảnh gốc theo tứ giác tìm được.
    Hỗ trợ phóng to độ phân giải ảnh đích (upscale_factor).
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Áp dụng nội suy phóng to
    if upscale_factor != 1.0:
        maxWidth = int(maxWidth * upscale_factor)
        maxHeight = int(maxHeight * upscale_factor)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    # Dùng LANCZOS4 nếu phóng to để nét chữ mượt và nét viền mềm mại hơn
    interp_flag = cv2.INTER_LANCZOS4 if upscale_factor > 1.0 else cv2.INTER_CUBIC
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=interp_flag)

    return warped, M

def map_boxes_to_original(boxes, transform_matrix):
    """
    Map danh sách các bounding boxes từ ảnh đã xử lý (warped) 
    về tọa độ của ảnh gốc (original).
    
    Args:
        boxes (list): Danh sách các boxes, mỗi box là list/array các điểm [[x,y],...].
        transform_matrix (numpy.ndarray): Ma trận biến đổi M (3x3).
        
    Returns:
        list: Danh sách các boxes đã được map về tọa độ gốc.
    """
    if transform_matrix is None or len(boxes) == 0:
        return [np.array(box).tolist() for box in boxes]
        
    M_inv = np.linalg.inv(transform_matrix)
    
    mapped_boxes = []
    for box in boxes:
        # Chuyển box về định dạng float32 (N, 1, 2) cho cv2.perspectiveTransform
        pts = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
        mapped_pts = cv2.perspectiveTransform(pts, M_inv)
        # Reshape lại thành (N, 2) và chuyển sang list int
        mapped_pts = mapped_pts.reshape(-1, 2).astype(np.int32).tolist()
        mapped_boxes.append(mapped_pts)
        
    return mapped_boxes

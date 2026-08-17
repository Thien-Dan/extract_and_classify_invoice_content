from typing import List, Dict, Any, Tuple
from .types import LayoutLMToken

def normalize_bbox(box: List[List[float]], width: int, height: int) -> List[int]:
    """
    Chuẩn hóa bounding box về tỷ lệ [0, 1000] của mô hình LayoutLM.
    
    Args:
        box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        width: Chiều rộng ảnh gốc
        height: Chiều cao ảnh gốc
        
    Returns:
        [x0, y0, x1, y1] đã được ép về số nguyên từ 0 đến 1000.
    """
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    
    # Lấy min, max
    x0, x1 = min(x_coords), max(x_coords)
    y0, y1 = min(y_coords), max(y_coords)
    
    # Scale về 1000 và kẹp biên (clip)
    norm_x0 = int(1000 * (x0 / width))
    norm_x1 = int(1000 * (x1 / width))
    norm_y0 = int(1000 * (y0 / height))
    norm_y1 = int(1000 * (y1 / height))
    
    # Đảm bảo không vượt quá [0, 1000]
    return [
        max(0, min(1000, norm_x0)),
        max(0, min(1000, norm_y0)),
        max(0, min(1000, norm_x1)),
        max(0, min(1000, norm_y1))
    ]

def prepare_layoutlm_data(boxes: List[List[List[float]]], texts: List[str], image_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Đóng gói dữ liệu hộp chữ và text vào cấu trúc JSON để nạp vào LayoutLM.
    
    Args:
        boxes: List of boxes.
        texts: List of strings.
        image_shape: (height, width) của ảnh.
        
    Returns:
        Một Dictionary chứa "words" và "bboxes" chuẩn format LayoutLM / FUNSD.
    """
    img_h, img_w = image_shape[:2]
    
    words = []
    normalized_boxes = []
    
    for box, text in zip(boxes, texts):
        if not text.strip():
            continue
            
        words.append(text)
        normalized_boxes.append(normalize_bbox(box, img_w, img_h))
        
    return {
        "words": words,
        "bboxes": normalized_boxes
    }

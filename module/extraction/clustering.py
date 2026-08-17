import numpy as np
from typing import List, Tuple
from .types import TextLine

def get_y_center(box: List[List[float]]) -> float:
    """Tính trung bình tọa độ Y của 1 box."""
    y_coords = [p[1] for p in box]
    return sum(y_coords) / len(y_coords)

def get_x_min(box: List[List[float]]) -> float:
    """Lấy tọa độ X nhỏ nhất (trái cùng) của 1 box."""
    return min(p[0] for p in box)
    
def get_box_height(box: List[List[float]]) -> float:
    """Tính chiều cao của 1 box."""
    y_coords = [p[1] for p in box]
    return max(y_coords) - min(y_coords)

def sort_and_group_boxes(boxes: List[List[List[float]]], texts: List[str], line_margin_ratio: float = 0.5) -> List[TextLine]:
    """
    Gom cụm các boxes thành từng dòng (TextLine) dựa trên tọa độ Y,
    sau đó sắp xếp các chữ trong cùng một dòng từ trái sang phải.
    
    Args:
        boxes: List of boxes, mỗi box là [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        texts: List of strings tương ứng với boxes
        line_margin_ratio: Độ lệch Y cho phép để coi là cùng 1 dòng (tỷ lệ so với chiều cao chữ)
        
    Returns:
        Danh sách các TextLine đã được sắp xếp từ trên xuống dưới.
    """
    if not boxes or not texts or len(boxes) != len(texts):
        return []

    # B1: Tính thông số cho từng box
    box_info = []
    for i, (box, text) in enumerate(zip(boxes, texts)):
        y_c = get_y_center(box)
        x_m = get_x_min(box)
        h = get_box_height(box)
        box_info.append({
            'index': i,
            'text': text,
            'box': box,
            'y_center': y_c,
            'x_min': x_m,
            'height': h
        })

    # B2: Sắp xếp theo y_center từ trên xuống dưới
    box_info.sort(key=lambda item: item['y_center'])

    # B3: Gom cụm thành các dòng
    lines = []
    current_line = []
    current_line_y = box_info[0]['y_center']
    current_line_h = box_info[0]['height']

    for item in box_info:
        y_c = item['y_center']
        
        # Nếu box này nằm trên cùng 1 dòng (độ lệch y_center nhỏ hơn margin)
        margin = current_line_h * line_margin_ratio
        if abs(y_c - current_line_y) <= margin:
            current_line.append(item)
            # Cập nhật lại y_center trung bình của dòng
            current_line_y = sum(b['y_center'] for b in current_line) / len(current_line)
        else:
            # Lưu dòng hiện tại
            lines.append(current_line)
            # Bắt đầu dòng mới
            current_line = [item]
            current_line_y = y_c
            current_line_h = item['height']

    if current_line:
        lines.append(current_line)

    # B4: Sắp xếp các box trong cùng một dòng từ trái qua phải (x_min)
    text_lines = []
    for line in lines:
        line.sort(key=lambda item: item['x_min'])
        
        full_text = " ".join([item['text'] for item in line])
        y_center_avg = sum([item['y_center'] for item in line]) / len(line)
        raw_texts = [item['text'] for item in line]
        raw_boxes = [item['box'] for item in line]
        
        text_lines.append(TextLine(
            text=full_text,
            y_center=y_center_avg,
            raw_texts=raw_texts,
            raw_boxes=raw_boxes
        ))

    return text_lines

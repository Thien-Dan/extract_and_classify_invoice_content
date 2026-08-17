from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class LayoutLMToken:
    """
    Biểu diễn một token/word để feed vào LayoutLM.
    Tọa độ box đã được chuẩn hóa về scale [0, 1000].
    """
    text: str
    box: List[int]  # [x0, y0, x1, y1] normalized

@dataclass
class TextLine:
    """
    Biểu diễn một dòng văn bản được gom cụm từ các bounding boxes.
    """
    text: str
    y_center: float
    raw_texts: List[str]
    raw_boxes: List[List[List[float]]]

@dataclass
class InvoiceExtractionResult:
    """
    Kết quả trích xuất cuối cùng.
    """
    date: Optional[str] = None
    layoutlm_data: Dict[str, Any] = field(default_factory=dict)
    
    # Những trường dưới đây sẽ do LayoutLM điền vào sau này
    total_amount: Optional[str] = None
    items: List[Dict[str, str]] = field(default_factory=list)

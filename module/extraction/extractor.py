from typing import List, Tuple
from .types import InvoiceExtractionResult
from .clustering import sort_and_group_boxes
from .date_parser import extract_date
from .layoutlm_formatter import prepare_layoutlm_data

class InformationExtractor:
    def __init__(self):
        """Khởi tạo module trích xuất thông tin (Heuristic + LayoutLM prep)."""
        pass
        
    def extract(self, boxes: List[List[List[float]]], texts: List[str], image_shape: Tuple[int, int]) -> InvoiceExtractionResult:
        """
        Trích xuất thông tin từ Raw Text OCR.
        
        Args:
            boxes: Danh sách tọa độ bounding boxes.
            texts: Danh sách chuỗi text tương ứng.
            image_shape: Shape của ảnh gốc (height, width, channels).
            
        Returns:
            InvoiceExtractionResult chứa Ngày mua (dùng Regex) và data chuẩn bị cho LayoutLM.
        """
        result = InvoiceExtractionResult()
        
        # 1. Gom các text rời rạc thành từng Hàng (Row/Line)
        # line_margin_ratio=0.5 nghĩa là nếu 2 cụm chữ có tâm Y lệch nhau không quá 50% chiều cao chữ thì coi như cùng 1 hàng
        lines = sort_and_group_boxes(boxes, texts, line_margin_ratio=0.5)
        
        # 2. Dùng Regex để tìm Ngày trên các dòng đã gom
        parsed_date = extract_date(lines)
        result.date = parsed_date
        
        # 3. Chuẩn hóa Bounding Box và đóng gói JSON chuẩn bị cho LayoutLM
        # Truyền image_shape (height, width)
        layoutlm_data = prepare_layoutlm_data(boxes, texts, image_shape)
        result.layoutlm_data = layoutlm_data
        
        return result

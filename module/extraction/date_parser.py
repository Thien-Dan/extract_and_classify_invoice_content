import re
from typing import List, Optional
from .types import TextLine

# Các pattern Regex bắt ngày tháng
DATE_PATTERNS = [
    # Match: Ngày 15/08/2020, Ngày: 15-08-20, 15/08/2020
    r"(?i)(?:ngày|date)?\s*[:\-\s]?\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})",
]

# Các pattern Regex bắt thời gian
TIME_PATTERNS = [
    # Match: 10:08, 14:30, 23:59 (Bỏ qua mili giây hoặc giây nếu không cần)
    r"(?i)(?:thời gian|time|giờ|gio)?\s*[:\-\s]?\s*([0-2]?[0-9]):([0-5][0-9])",
]

def extract_date(lines: List[TextLine]) -> Optional[str]:
    """
    Duyệt qua các dòng text và dùng Regex để tìm Ngày và Giờ mua.
    """
    found_date = None
    found_time = None
    
    for line in lines:
        text = line.text
        
        # Tìm Ngày
        if not found_date:
            for pattern in DATE_PATTERNS:
                match = re.search(pattern, text)
                if match:
                    # Trích xuất ra ngày tháng năm định dạng chuẩn DD/MM/YYYY
                    day = match.group(1).zfill(2)
                    month = match.group(2).zfill(2)
                    year = match.group(3)
                    if len(year) == 2:
                        year = "20" + year
                    found_date = f"{day}/{month}/{year}"
                    break
                    
        # Tìm Giờ
        if not found_time:
            for pattern in TIME_PATTERNS:
                match = re.search(pattern, text)
                if match:
                    hour = match.group(1).zfill(2)
                    minute = match.group(2)
                    found_time = f"{hour}:{minute}"
                    break
                    
    # Kết hợp kết quả
    if found_date and found_time:
        return f"{found_date} {found_time}"
    elif found_date:
        return found_date
    return None

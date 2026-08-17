STORES = [
    "Vinmart", "Vinmart+", "Circle K", "Bách Hóa Xanh", "Co.op Food", 
    "Co.op Mart", "Lotte Mart", "GS25", "FamilyMart", "Ministop", 
    "7-Eleven", "Aeon Mall", "Big C", "GO!", "Emart", "Mega Market",
    "WinMart", "WinMart+", "K-Market", "G-Mart", "Thực phẩm sạch Sói Biển",
    "Guardian", "Pharmacity", "Long Châu", "Bibo Mart", "Kids Plaza"
]

# 1. NÂNG CẤP ITEMS: Thêm đồ siêu dài (ép rớt dòng) và đồ khuyến mãi
ITEMS = [
    # Đồ siêu dài (Để sinh ra Multi-line Items)
    "Nước giặt OMO Matic hương ngàn hoa túi zip 3.6kg tặng kèm 1 chai nước xả",
    "Sữa bột Ensure Gold hương vani hộp thiếc 850g (Dành cho người lớn tuổi)",
    "Bánh Trung Thu Kinh Đô hộp 4 cái x 150g (Thập cẩm, Đậu xanh, Sữa dừa, Khoai môn)",
    "Combo 2 chai tương ớt Chinsu 250g + 1 Nước mắm Nam Ngư 500ml",
    
    # Đồ bình thường
    "Sữa tươi Vinamilk 180ml", "Pepsi vị chanh không calo", "Bia Heineken lon 330ml",
    "Mì Hảo Hảo chua cay", "Snack khoai tây Lay's", "Xúc xích Vissan",
    
    # Hàng nhiễu (Dễ làm mô hình nhầm lẫn)
    "[KM] Tặng 1 bát sứ", 
    "-> Chiết khấu khách hàng thân thiết",
    "Mã giảm giá thanh toán VNPay"
]

# 2. BỔ SUNG CÁC TRƯỜNG ĐỊNH LƯỢNG (Rất quan trọng cho LayoutLM)
QUANTITIES = [
    "1", "2", "3", "5", "10", 
    "1.5", "0.5", # Cân ký (ví dụ: 1.5 kg)
    "01", "02"    # Máy in bill hay có số 0 ở đầu
]

PRICES = [
    # Có ký hiệu
    "150,000 VND", "35.000đ", "1,500,000 đ", "5.000 VNĐ",
    # Không ký hiệu, phân cách dấu chấm/phẩy (Việt Nam hay dùng lẫn lộn)
    "150.000", "150,000", "2.500.000", "12,500",
    # Lỗi OCR / Máy in mờ mất dấu
    "150000", "35000", "150 000"
]

# 3. NÂNG CẤP DATES: Thêm nhiều chuẩn định dạng và Lỗi
DATES = [
    f"{d:02d}/{m:02d}/{y}" for y in range(2021, 2024) for m in range(1, 12) for d in range(1, 28)
] + [
    # Đổi dấu phân cách
    "15.08.2023", "15-08-2023", "15/08/23",
    # Text dài
    "Ngày 15 tháng 08 năm 2023", 
    "Ngày bán: 15/08/2023 14:30:00",
    # Lỗi OCR
    "Ngay ban: 15/08/2023", "Date: 15-08-2023"
]

ADDRESSES = [
    "Đ/c: Số 1, Đại Cồ Việt, Hai Bà Trưng, HN",
    "Tổ 7, Khu Minh Tiến Anh",
    "P. Cẩm Bình, TP. Cẩm Phả, QN",
    "234 Nguyễn Trãi, Thanh Xuân, Hà Nội",
    "Khu dân cư Chánh Nghĩa, Thủ Dầu Một, Bình Dương",
    "Tầng 1 TTTM Aeon Mall, Long Biên, Hà Nội"
]

PHONES = [
    "Hotline: 1900 1567",
    "SĐT: 02471086386",
    "Hotline: 028 3930 4567",
    "Điện thoại: 0988 123 456"
]

# 4. NÂNG CẤP HEADERS/FOOTERS: Thêm nhiễu OCR
HEADERS = [
    "HÓA ĐƠN BÁN HÀNG",
    "HÓA DƠN BÁN HÀN", # Lỗi OCR
    "PHIẾU THANH TOÁN",
    "PHIEU THANH TOAN", # Không dấu
    "HÓA ĐƠN GTGT",
    "BIÊN LAI BÁN LẺ",
    "*** RECEIPT ***"
]

FOOTERS = [
    "Giá đã bao gồm thuế GTGT",
    "Gia da bao gom thue", # Không dấu
    "CÁM ƠN QUÝ KHÁCH VÀ HẸN GẶP LẠI",
    "CAM ƠN QUY KHACH", # Lỗi OCR dấu
    "Wifi: vinmart_free / Pass: 12345678", # Text rác hay xuất hiện ở cuối bill
    "Chỉ xuất hóa đơn trong ngày",
    "Tax invoice will be issued within same day",
    "Website: vinmart.com"
]

BARCODES = [
    "89360132339185",
    "038500075117",
    "8938502390123",
    "|| ||| | || ||| || ||",
    "* 1 2 3 4 5 6 7 8 9 *"
]

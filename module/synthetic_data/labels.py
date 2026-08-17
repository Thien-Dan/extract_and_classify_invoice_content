# module/synthetic_data/labels.py

class LayoutLMLabels:
    """
    Định nghĩa hệ thống nhãn BIO cho bài toán trích xuất hóa đơn.
    """
    O = "O"
    
    # Store name
    B_STORE = "B-STORE"
    I_STORE = "I-STORE"
    
    # Date
    B_DATE = "B-DATE"
    I_DATE = "I-DATE"
    
    # Item name
    B_ITEM = "B-ITEM"
    I_ITEM = "I-ITEM"
    
    # Quantity
    B_QTY = "B-QTY"
    I_QTY = "I-QTY"
    
    # Price
    B_PRICE = "B-PRICE"
    I_PRICE = "I-PRICE"
    
    # Total
    B_TOTAL = "B-TOTAL"
    I_TOTAL = "I-TOTAL"

    @classmethod
    def get_all_labels(cls):
        return [
            cls.O, 
            cls.B_STORE, cls.I_STORE, 
            cls.B_DATE, cls.I_DATE,
            cls.B_ITEM, cls.I_ITEM,
            cls.B_QTY, cls.I_QTY,
            cls.B_PRICE, cls.I_PRICE,
            cls.B_TOTAL, cls.I_TOTAL
        ]

class LayoutLMItemLabels:
    O = "O"
    B_ITEM = "B-ITEM"
    I_ITEM = "I-ITEM"
    B_QTY = "B-QTY"
    I_QTY = "I-QTY"
    B_PRICE = "B-PRICE"
    I_PRICE = "I-PRICE"
    
    @classmethod
    def get_all_labels(cls):
        return [
            cls.O, 
            cls.B_ITEM, cls.I_ITEM,
            cls.B_QTY, cls.I_QTY,
            cls.B_PRICE, cls.I_PRICE
        ]

import os
import random
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from .labels_v3 import LayoutLMv3Labels
from .vocab import STORES, ITEMS, DATES, ADDRESSES, PHONES, HEADERS, FOOTERS, BARCODES, QUANTITIES, PRICES

class ReceiptGenerator:
    def __init__(self, font_path="RobotoMono-Regular.ttf"):
        self.font_path = font_path
        self._ensure_font_exists()
        
    def _ensure_font_exists(self):
        if not os.path.exists(self.font_path):
            print(f"Downloading font {self.font_path}...")
            url = "https://github.com/googlefonts/RobotoMono/raw/main/fonts/ttf/RobotoMono-Regular.ttf"
            urllib.request.urlretrieve(url, self.font_path)
            
    def _get_random_font(self):
        size = random.randint(18, 28)
        return ImageFont.truetype(self.font_path, size)
            
    def _normalize_bbox(self, box, width, height):
        x0, y0, x1, y1 = box
        norm_x0 = int(1000 * (x0 / width))
        norm_y0 = int(1000 * (y0 / height))
        norm_x1 = int(1000 * (x1 / width))
        norm_y1 = int(1000 * (y1 / height))
        return [
            max(0, min(1000, norm_x0)),
            max(0, min(1000, norm_y0)),
            max(0, min(1000, norm_x1)),
            max(0, min(1000, norm_y1))
        ]

    def _draw_text_and_label(self, draw, text, position, label_b, label_i, words, bboxes, labels, img_w, img_h, font=None, update_y=True, max_width=None):
        if font is None:
            font = self._get_random_font()
            
        x, y = position
        start_x = x
        max_y = y
        
        if random.random() > 0.5 and '.' in text and any(c.isdigit() for c in text):
            tokens = text.split('.')
            tokens = [tokens[0]] + ['.' + t for t in tokens[1:]]
        else:
            tokens = text.split()
            
        for i, token in enumerate(tokens):
            if not token.strip(): continue
            
            bbox = draw.textbbox((x, y), token, font=font)
            token_w = bbox[2] - bbox[0]
            space_w = draw.textlength(" ", font=font)
            
            if max_width is not None and (x + token_w > max_width):
                x = start_x
                y += int(font.size * 1.3)
                bbox = draw.textbbox((x, y), token, font=font)
                
            draw.text((x, y), token, fill="black", font=font)
            
            if bbox[3] > max_y:
                max_y = bbox[3]
                
            x += token_w + space_w + random.randint(-2, 5)
            
            current_label = label_b if i == 0 else label_i
            words.append(token)
            bboxes.append(self._normalize_bbox(bbox, img_w, img_h))
            labels.append(current_label)
            
        return max_y if update_y else x

    def generate_receipt(self):
        layout_style = random.choice(["Table", "POS", "Card"])
        num_items = random.randint(1, 30)
        
        if layout_style == "POS":
            width = random.randint(400, 500)
        else:
            width = random.randint(700, 1000)
            
        height = num_items * 150 + 700
        image = Image.new('RGB', (width, height), "white")
        draw = ImageDraw.Draw(image)
        
        words = []
        bboxes = []
        labels = []
        
        current_y = random.randint(20, 50)
        
        store = random.choice(STORES)
        current_y = self._draw_text_and_label(draw, store, (random.randint(50, 150), current_y), LayoutLMv3Labels.B_SELLER, LayoutLMv3Labels.I_SELLER, words, bboxes, labels, width, height) + random.randint(25, 45)
        
        address = random.choice(ADDRESSES)
        current_y = self._draw_text_and_label(draw, address, (random.randint(50, 150), current_y), LayoutLMv3Labels.B_ADDRESS, LayoutLMv3Labels.I_ADDRESS, words, bboxes, labels, width, height) + random.randint(25, 45)
        
        date = random.choice(DATES)
        current_y = self._draw_text_and_label(draw, date, (random.randint(50, 150), current_y), LayoutLMv3Labels.B_TIMESTAMP, LayoutLMv3Labels.I_TIMESTAMP, words, bboxes, labels, width, height) + random.randint(25, 45)
        
        current_y += 30
        item_font = self._get_random_font()
        
        if layout_style == "Table":
            self._draw_text_and_label(draw, "Tên Hàng", (10, current_y), LayoutLMv3Labels.O, LayoutLMv3Labels.O, words, bboxes, labels, width, height, font=item_font, update_y=False)
            self._draw_text_and_label(draw, "SL", (width - 350, current_y), LayoutLMv3Labels.O, LayoutLMv3Labels.O, words, bboxes, labels, width, height, font=item_font, update_y=False)
            self._draw_text_and_label(draw, "Đơn Giá", (width - 250, current_y), LayoutLMv3Labels.O, LayoutLMv3Labels.O, words, bboxes, labels, width, height, font=item_font, update_y=False)
            self._draw_text_and_label(draw, "Thành Tiền", (width - 120, current_y), LayoutLMv3Labels.O, LayoutLMv3Labels.O, words, bboxes, labels, width, height, font=item_font, update_y=False)
            current_y += 50
            
            for _ in range(num_items):
                item_name = random.choice(ITEMS)
                qty = random.choice(QUANTITIES)
                price = random.choice(PRICES)
                
                # Check length for table wrap
                if len(item_name) > 25:
                    max_y = self._draw_text_and_label(draw, item_name, (10, current_y), LayoutLMv3Labels.B_ITEM, LayoutLMv3Labels.I_ITEM, words, bboxes, labels, width, height, font=item_font, update_y=True, max_width=width-400)
                    self._draw_text_and_label(draw, qty, (width - 350, current_y), LayoutLMv3Labels.B_QTY, LayoutLMv3Labels.I_QTY, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, price, (width - 250, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, price, (width - 120, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    current_y = max_y + random.randint(40, 60)
                else:
                    self._draw_text_and_label(draw, item_name, (10, current_y), LayoutLMv3Labels.B_ITEM, LayoutLMv3Labels.I_ITEM, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, qty, (width - 350, current_y), LayoutLMv3Labels.B_QTY, LayoutLMv3Labels.I_QTY, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, price, (width - 250, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, price, (width - 120, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    current_y += random.randint(40, 60)
                
        elif layout_style == "POS":
            for _ in range(num_items):
                item_name = random.choice(ITEMS)
                qty = random.choice(QUANTITIES)
                price = random.choice(PRICES)
                
                max_y = self._draw_text_and_label(draw, item_name, (10, current_y), LayoutLMv3Labels.B_ITEM, LayoutLMv3Labels.I_ITEM, words, bboxes, labels, width, height, font=item_font, update_y=True, max_width=width-20)
                current_y = max_y + random.randint(35, 45)
                
                nx = 20
                nx = self._draw_text_and_label(draw, qty, (nx, current_y), LayoutLMv3Labels.B_QTY, LayoutLMv3Labels.I_QTY, words, bboxes, labels, width, height, font=item_font, update_y=False)
                nx = self._draw_text_and_label(draw, "x", (nx + 10, current_y), LayoutLMv3Labels.O, LayoutLMv3Labels.O, words, bboxes, labels, width, height, font=item_font, update_y=False)
                nx = self._draw_text_and_label(draw, price, (nx + 10, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                
                self._draw_text_and_label(draw, price, (width - 120, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                current_y += random.randint(45, 60)
                
        elif layout_style == "Card":
            for i in range(num_items):
                item_name = random.choice(ITEMS)
                qty = random.choice(QUANTITIES)
                price = random.choice(PRICES)
                
                if random.random() > 0.5:
                    max_y = self._draw_text_and_label(draw, item_name, (random.randint(10, 50), current_y), LayoutLMv3Labels.B_ITEM, LayoutLMv3Labels.I_ITEM, words, bboxes, labels, width, height, font=item_font, update_y=True, max_width=width-300)
                    nx = self._draw_text_and_label(draw, qty, (width - 250, current_y), LayoutLMv3Labels.B_QTY, LayoutLMv3Labels.I_QTY, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, price, (nx + 20, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    current_y = max(max_y, current_y) + random.randint(40, 60)
                else:
                    nx = self._draw_text_and_label(draw, qty, (10, current_y), LayoutLMv3Labels.B_QTY, LayoutLMv3Labels.I_QTY, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    nx = self._draw_text_and_label(draw, item_name, (nx + 20, current_y), LayoutLMv3Labels.B_ITEM, LayoutLMv3Labels.I_ITEM, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    self._draw_text_and_label(draw, price, (nx + 20, current_y), LayoutLMv3Labels.B_PRICE, LayoutLMv3Labels.I_PRICE, words, bboxes, labels, width, height, font=item_font, update_y=False)
                    current_y += random.randint(40, 60)
                
        current_y += 30
        self._draw_text_and_label(draw, "TỔNG CỘNG:", (10, current_y), LayoutLMv3Labels.O, LayoutLMv3Labels.O, words, bboxes, labels, width, height, font=item_font, update_y=False)
        self._draw_text_and_label(draw, random.choice(PRICES), (width - 200, current_y), LayoutLMv3Labels.B_TOTAL_COST, LayoutLMv3Labels.I_TOTAL_COST, words, bboxes, labels, width, height, font=item_font, update_y=False)
        current_y += 50
        
        final_image = image.crop((0, 0, width, current_y + 50))
        
        adjusted_bboxes = []
        for box in bboxes:
            orig_height = height
            new_height = current_y + 50
            
            y0 = box[1] * orig_height / 1000.0
            y1 = box[3] * orig_height / 1000.0
            
            norm_y0 = int(1000 * (y0 / new_height))
            norm_y1 = int(1000 * (y1 / new_height))
            
            adjusted_bboxes.append([box[0], norm_y0, box[2], norm_y1])
        
        return final_image, {
            "words": words,
            "bboxes": adjusted_bboxes,
            "labels": labels
        }

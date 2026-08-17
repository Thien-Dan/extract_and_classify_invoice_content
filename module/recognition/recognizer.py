import os
import torch
import numpy as np
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class TextRecognizer:
    def __init__(self, config_name='vgg_transformer', device='cpu'):
        """
        Khởi tạo module nhận dạng chữ.
        
        Args:
            config_name (str): Tên cấu hình của VietOCR ('vgg_transformer', 'resnet_transformer', 'vgg_seq2seq').
            device (str): Nơi chạy mô hình ('cpu', 'cuda:0').
        """
        config = Cfg.load_config_from_name(config_name)
        config['device'] = device
        
        # Kiểm tra xem có mô hình fine-tune không
        finetuned_weights = 'saved_models/vietocr_mcocr/weights.pth'
        if os.path.exists(finetuned_weights):
            print(f"--- Đã tìm thấy mô hình VietOCR mài giũa (Fine-tuned), đang tải từ {finetuned_weights} ---")
            config['weights'] = finetuned_weights
        
        # Bật Beam Search theo yêu cầu để giữ độ chính xác cao nhất
        config['predictor']['beamsearch'] = True
        
        # Tăng image_max_width để tránh bóp méo (nén) các khung chữ quá dài của hóa đơn
        config['dataset']['image_max_width'] = 2048
        
        if not os.path.exists(finetuned_weights):
            print(f"--- Đang tải mô hình VietOCR mặc định ({config_name}, BeamSearch=True, BeamSize=5) trên {device} ---")
            
        self.detector = Predictor(config)
        
        # [MẸO NHỎ] Thư viện VietOCR mặc định fix cứng beam_size=4 và không đọc từ config.
        # Để thay đổi beam_size, ta dùng kỹ thuật Monkey-patching ghi đè hàm mặc định của nó:
        import vietocr.tool.predictor
        from vietocr.tool.translate import translate_beam_search
        import functools
        
        # Ghi đè hàm translate_beam_search cục bộ để ép nó chạy với beam_size=5
        vietocr.tool.predictor.translate_beam_search = functools.partial(translate_beam_search, beam_size=5)
        
    def preprocess_image(self, img, padding=4):
        """
        Tiền xử lý ảnh chuyên sâu cho VietOCR.
        Bao gồm: Padding (tạo viền trắng), Sharpening (làm nét), và CLAHE (cân bằng sáng).
        """
        if img is None or img.size == 0:
            return img
            
        import cv2
        # 1. Chuyển sang ảnh xám để triệt tiêu nhiễu màu nền
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. Tăng cường độ tương phản (CLAHE) để chữ sắc nét và nổi bật khỏi nền
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Chuyển ngược lại BGR (vì lát nữa code gọi cvtColor sang RGB)
        img_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return img_bgr

    def recognize(self, images, return_prob=True):
        """
        Nhận dạng chữ trong danh sách các ảnh.
        
        Args:
            images (list): Danh sách các ảnh con đã cắt (định dạng numpy.ndarray BGR).
            return_prob (bool): Trả về kết quả kèm độ tin cậy (confidence score) nếu True.
            
        Returns:
            list: Danh sách kết quả dạng text hoặc dạng tuple (text, prob).
        """
        if not images:
            return []
            
        import cv2
        
        pil_images = []
        for img in images:
            # Nếu ảnh nhỏ hoặc lỗi
            if img.shape[0] == 0 or img.shape[1] == 0:
                pil_images.append(Image.new('RGB', (10, 10), (255, 255, 255)))
                continue
                
            # --- Tiền xử lý chữ (Text-line Preprocessing) ---
            img_bgr = self.preprocess_image(img, padding=4)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img_rgb)
            pil_images.append(pil_img)
            
        results = []
        # Chạy từng ảnh một (không dùng predict_batch) để tránh bị chèn padding đen 
        # làm méo mó các ảnh có chiều ngang (aspect ratio) khác biệt nhau quá lớn.
        for img in pil_images:
            try:
                res = self.detector.predict(img, return_prob=return_prob)
                results.append(res)
            except Exception as e:
                print(f"Lỗi khi nhận dạng ảnh đơn: {e}")
                results.append(("", 0.0) if return_prob else "")
                
        return results

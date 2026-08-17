import os
import sys
import logging
import cv2
import numpy as np
from .utils import get_rotated_crop

# Tắt bớt log rác từ PaddleOCR
logging.getLogger("ppocr").setLevel(logging.WARNING)

class DBNetDetector:
    "\"\"
    Module nhận diện vùng văn bản (Text Detection) sử dụng kiến trúc DBNet (PaddleOCR/PaddleX).
    Được gói gọn để dễ dàng tích hợp vào hệ thống Backend (Sever) hoặc luồng Pipeline.
    "\"\"
    
    def __init__(self, lang='en', use_gpu=False):
        "\"\"
        Khởi tạo mô hình DBNet.
        "\"\"
        self.use_gpu = use_gpu
        self.lang = lang
        
        import os
        model_dir = os.path.join(os.path.dirname(__file__), 'finetuned_dbnet')
        
        # Dùng trực tiếp TextDetRunnerPredictor của PaddleX
        from paddlex.inference.models import create_predictor
        self.model = create_predictor(model_name="PP-OCRv4_mobile_det", model_dir=model_dir)
        
        from module.preprocessing import CropConfig
        self.crop_config = CropConfig()

    def detect(self, image, use_preprocessor=True, enable_tiling=True, tile_overlap=300):
        if use_preprocessor:
            from module.preprocessing import detect_and_crop_receipt
            result = detect_and_crop_receipt(image, cfg=self.crop_config)
            if result.success and result.warped_image is not None:
                image = result.warped_image
                self.last_metadata = {"transform_matrix": result.transform_matrix}
            else:
                if isinstance(image, str):
                    image = cv2.imread(image)
                self.last_metadata = None
            self.last_processed_image = image
        else:
            if isinstance(image, str):
                image = cv2.imread(image)
            self.last_processed_image = image
            self.last_metadata = None

        h, w = image.shape[:2]
        boxes = []
        
        if enable_tiling and h > w * 1.5:
            tile_size = w
            stride = tile_size - tile_overlap
            if stride <= 0:
                stride = tile_size // 2
                
            y = 0
            while y < h:
                y_end = min(y + tile_size, h)
                tile = image[y:y_end, :]
                
                results = list(self.model(tile))
                if results and len(results) > 0:
                    res = results[0]
                    if 'dt_polys' in res and len(res['dt_polys']) > 0:
                        tile_boxes = res['dt_polys']
                        for b in tile_boxes:
                            for pt in b:
                                pt[1] += y
                        boxes.extend(tile_boxes)
                
                if y_end == h:
                    break
                y += stride
                
            from module.DBnet.nms import poly_nms
            boxes = poly_nms(boxes, iou_threshold=0.3)
        else:
            results = list(self.model(image))
            if results and len(results) > 0:
                res = results[0]
                if 'dt_polys' in res and len(res['dt_polys']) > 0:
                    boxes = res['dt_polys']
        
        if use_preprocessor:
            from module.preprocessing import map_boxes_to_original
            self._last_boxes_before_mapping = [np.array(b).tolist() for b in boxes]
            boxes = map_boxes_to_original(boxes, self.last_metadata["transform_matrix"])
        else:
            boxes = [np.array(b).tolist() for b in boxes]
            self._last_boxes_before_mapping = boxes

        boxes = sorted(boxes, key=lambda b: b[0][1])
        return boxes
    
    def get_crops(self, image, boxes, margin_x=4, margin_y=1):
        if margin_x > 0 or margin_y > 0:
            boxes_to_crop = self.expand_boxes(boxes, image.shape, margin_x, margin_y)
        else:
            boxes_to_crop = boxes
            
        crops = []
        for box in boxes_to_crop:
            cropped = get_rotated_crop(image, box)
            crops.append(cropped)
        return crops

    def expand_boxes(self, boxes, img_shape, margin_x, margin_y):
        expanded_boxes = []
        h_img, w_img = img_shape[:2]
        for box in boxes:
            poly = np.array(box, dtype=np.float32)
            center = np.mean(poly, axis=0)
            expanded_box = []
            for pt in poly:
                vec = pt - center
                dx = margin_x if vec[0] > 0 else -margin_x
                dy = margin_y if vec[1] > 0 else -margin_y
                new_x = max(0, min(w_img - 1, pt[0] + dx))
                new_y = max(0, min(h_img - 1, pt[1] + dy))
                expanded_box.append([new_x, new_y])
            expanded_boxes.append(expanded_box)
        return expanded_boxes

    def draw_boxes(self, image, boxes, color=(0, 255, 0), thickness=2):
        img_copy = image.copy()
        for box in boxes:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
        return img_copy

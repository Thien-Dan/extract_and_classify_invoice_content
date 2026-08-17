import os
import sys
import logging
import cv2
import numpy as np
from .utils import get_rotated_crop

# Tắt bớt log rác từ PaddleOCR
logging.getLogger("ppocr").setLevel(logging.WARNING)

class DictArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class DBNetDetector:
    """
    Module nhận diện vùng văn bản (Text Detection) sử dụng kiến trúc DBNet (ONNX Runtime).
    Được gói gọn để dễ dàng tích hợp vào hệ thống Backend (Sever) hoặc luồng Pipeline.
    """
    
    def __init__(self, lang='en', use_gpu=False):
        """
        Khởi tạo mô hình DBNet.
        
        Args:
            lang (str): Ngôn ngữ của mô hình.
            use_gpu (bool): Bật/Tắt chế độ chạy GPU.
        """
        self.use_gpu = use_gpu
        self.lang = lang
        
        # Đường dẫn tới thư mục PaddleOCR để import các module post-process
        paddleocr_dir = os.path.join(os.path.dirname(__file__), 'PaddleOCR')
        if paddleocr_dir not in sys.path:
            sys.path.insert(0, paddleocr_dir)
            
        from tools.infer.predict_det import TextDetector
        
        model_dir = os.path.join(os.path.dirname(__file__), 'finetuned_dbnet', 'model.onnx')
        
        # Cấu hình giả lập args cho TextDetector của PaddleOCR
        # Sử dụng đúng 736 để map chuẩn 100% với PaddleX cũ
        args = DictArgs(
            use_onnx=True,
            det_model_dir=model_dir,
            det_algorithm='DB',
            det_limit_side_len=736,
            det_limit_type='max',
            det_box_type='quad',
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
            use_dilation=False,
            det_db_score_mode='fast',
            benchmark=False,
            use_gpu=use_gpu,
            det_pth_model_dir=None,
            page_num=0,
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            det_sast_score_thresh=0.5,
            det_sast_nms_thresh=0.2,
            det_pse_thresh=0,
            det_pse_box_thresh=0.85,
            det_pse_min_area=16,
            det_pse_scale=1,
            scales=[8, 16, 32],
            alpha=1.0,
            beta=1.0,
            fourier_degree=5,
            show_log=False
        )
        self.model = TextDetector(args)
        
        # Khởi tạo Preprocessor
        from module.preprocessing import CropConfig
        self.crop_config = CropConfig()

    def detect(self, image, use_preprocessor=True, enable_tiling=True, tile_overlap=300):
        """
        Detect text in the given image.
        """
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
                if image is None:
                    raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image}")
            self.last_processed_image = image
            self.last_metadata = None

        h, w = image.shape[:2]
        boxes = []
        
        # Nếu bật Tiling và ảnh có chiều cao lớn hơn 1.5 lần chiều rộng
        if enable_tiling and h > w * 1.5:
            print(f">> [Tiling] Ảnh dài (tỷ lệ {h/w:.1f}), tiến hành chia nhỏ...")
            tile_size = w
            stride = tile_size - tile_overlap
            if stride <= 0:
                stride = tile_size // 2
                
            y = 0
            while y < h:
                y_end = min(y + tile_size, h)
                tile = image[y:y_end, :]
                
                dt_boxes, _ = self.model(tile)
                if dt_boxes is not None and len(dt_boxes) > 0:
                    tile_boxes = dt_boxes.tolist()
                    # Shift y coordinates by offset (y)
                    for b in tile_boxes:
                        for pt in b:
                            pt[1] += y
                    boxes.extend(tile_boxes)
                
                if y_end == h:
                    break
                y += stride
                
            # Loại bỏ hộp trùng lặp (NMS)
            from module.DBnet.nms import poly_nms
            boxes = poly_nms(boxes, iou_threshold=0.3)
            print(f">> [Tiling] Đã gộp thành {len(boxes)} khung chữ.")
        else:
            # Chạy dự đoán bình thường (nguyên ảnh)
            dt_boxes, _ = self.model(image)
            if dt_boxes is not None and len(dt_boxes) > 0:
                boxes = dt_boxes.tolist()
        
        if use_preprocessor:
            from module.preprocessing import map_boxes_to_original
            # Lưu boxes ở tọa độ processed image trước khi map, dùng cho first-pass filter
            self._last_boxes_before_mapping = [np.array(b).tolist() for b in boxes]
            boxes = map_boxes_to_original(boxes, self.last_metadata["transform_matrix"])
        else:
            boxes = [np.array(b).tolist() for b in boxes]
            self._last_boxes_before_mapping = boxes


        # Sắp xếp từ trên xuống dưới (dựa vào tọa độ y của góc trên cùng bên trái)
        boxes = sorted(boxes, key=lambda b: b[0][1])
        
        return boxes
    
    def get_crops(self, image, boxes, margin_x=4, margin_y=1):
        """
        Cắt các vùng ảnh con dựa trên danh sách bounding boxes.
        """
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
        """
        Mở rộng bounding boxes thêm margin_x và margin_y.
        """
        expanded_boxes = []
        h_img, w_img = img_shape[:2]
        
        for box in boxes:
            poly = np.array(box, dtype=np.float32)
            # Tìm tâm của hộp để biết hướng mở rộng
            center = np.mean(poly, axis=0)
            expanded_box = []
            for pt in poly:
                vec = pt - center
                # Mở rộng ra xa tâm theo từng trục
                dx = margin_x if vec[0] > 0 else -margin_x
                dy = margin_y if vec[1] > 0 else -margin_y
                
                # Tránh mở rộng vượt quá kích thước ảnh
                new_x = max(0, min(w_img - 1, pt[0] + dx))
                new_y = max(0, min(h_img - 1, pt[1] + dy))
                expanded_box.append([new_x, new_y])
            expanded_boxes.append(expanded_box)
        return expanded_boxes


    def draw_boxes(self, image, boxes, color=(0, 255, 0), thickness=2):
        """
        Hỗ trợ trực quan hóa: Vẽ các bounding boxes lên ảnh gốc.
        """
        img_copy = image.copy()
        for box in boxes:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
        return img_copy

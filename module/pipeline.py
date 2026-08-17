import os
import os
import cv2
from typing import Dict, Any

from module.config import Config
from module.DBnet.detector import DBNetDetector
from module.recognition.recognizer import TextRecognizer
from module.preprocessing.deskew import ImageDeskewer
from module.extraction.extractor import InformationExtractor
from module.finetune.inferencer_v3_onnx import LayoutLMv3Inferencer
from module.preprocessing.pipeline import detect_and_crop_receipt

class InvoicePipeline:
    def __init__(self, device='cpu', skew_threshold=1.5):
        self.device = device
        self.skew_threshold = skew_threshold
        
        print("1. Loading DBNet Text Detector...")
        self.detector = DBNetDetector(use_gpu=('cuda' in self.device))
        
        print("2. Loading VietOCR Text Recognizer...")
        self.recognizer = TextRecognizer(config_name=Config.VIETOCR_CONFIG_NAME, device=self.device)
        
        print("3. Loading Image Deskewer & Extractor...")
        self.deskewer = ImageDeskewer(skew_threshold=self.skew_threshold)
        self.extractor = InformationExtractor()
        
        print("4. Loading LayoutLMv3 Inferencer (4 Labels)...")
        self.inferencer = LayoutLMv3Inferencer(model_dir="saved_models/layoutlmv3_onnx")
        
        print("=== PIPELINE READY ===\n")

    def extract_boxes_and_text(self, image_path):
        import numpy as np
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.imread(image_path)
        elif isinstance(image_path, np.ndarray):
            image = image_path.copy()
        else:
            raise ValueError("image_path must be a file path string or numpy array")
        
        from module.preprocessing.types import CropConfig
        prep_config = CropConfig()
        prep_result = detect_and_crop_receipt(image, output_path=None, cfg=prep_config)
        warped_img = prep_result.warped_image if prep_result.success else image
        
        initial_boxes = self.detector.detect(warped_img)
        corrected_img, angle = self.deskewer.deskew_and_rotate(warped_img, initial_boxes)
        
        if abs(angle) >= self.skew_threshold:
            boxes = self.detector.detect(corrected_img)
        else:
            boxes = initial_boxes
            
        crops = self.detector.get_crops(corrected_img, boxes)
        recognized_data = self.recognizer.recognize(crops, return_prob=False)
        
        return corrected_img, boxes, recognized_data

    def process(self, image_path, output_image_path=None):
        corrected_img, boxes, recognized_data = self.extract_boxes_and_text(image_path)
        
        extracted_data = self.extractor.extract(boxes, recognized_data, corrected_img.shape)
        layoutlm_input = extracted_data.layoutlm_data
        
        from PIL import Image; rgb_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB); pil_img = Image.fromarray(rgb_img); final_labels = self.inferencer.predict(pil_img, layoutlm_input["words"], layoutlm_input["bboxes"])
        
        results_list = []
        for w, bbox, label in zip(layoutlm_input["words"], layoutlm_input["bboxes"], final_labels):
            results_list.append({"text": w, "label": label, "bbox": bbox})
            
        extracted_items = []
        item_data = []
        for w, bbox, label in zip(layoutlm_input["words"], layoutlm_input["bboxes"], final_labels):
            if label in ["B-ITEM", "I-ITEM", "B-QTY", "I-QTY", "B-PRICE", "I-PRICE"]:
                y_center = (bbox[1] + bbox[3]) / 2.0
                item_data.append({"text": w, "bbox": bbox, "label": label, "y_center": y_center})
                
        item_data.sort(key=lambda x: x["y_center"])
        lines = []
        if item_data:
            current_line = [item_data[0]]
            for i in range(1, len(item_data)):
                if abs(item_data[i]["y_center"] - current_line[-1]["y_center"]) < 15:
                    current_line.append(item_data[i])
                else:
                    lines.append(current_line)
                    current_line = [item_data[i]]
            lines.append(current_line)
            
        for line in lines:
            line.sort(key=lambda x: x["bbox"][0])
            
            item_name, qty, price = [], [], []
            for token in line:
                l = token["label"]
                t = token["text"]
                if l in ["B-ITEM", "I-ITEM"]: item_name.append(t)
                elif l in ["B-QTY", "I-QTY"]: qty.append(t)
                elif l in ["B-PRICE", "I-PRICE"]: price.append(t)
                    
            if item_name or price or qty:
                extracted_items.append({
                    "item": " ".join(item_name),
                    "qty": " ".join(qty),
                    "price": " ".join(price)
                })

        if output_image_path:
            filtered_boxes = [box for box, text in zip(boxes, recognized_data) if text.strip()]
            self._draw_results(corrected_img, filtered_boxes, layoutlm_input["words"], final_labels, output_image_path)
            
        return {
            'regex_data': {
                'invoice_info': {'date': extracted_data.date}
            },
            'layoutlm_labels': results_list,
            'items': extracted_items,
            'processed_image_path': output_image_path
        }

    def _draw_results(self, image, boxes, words, labels_main, output_path):
        import numpy as np
        vis_img = image.copy()
        
        colors = {
            'B-SELLER': (0, 0, 255), 'I-SELLER': (0, 0, 255),
            'B-ADDRESS': (0, 255, 0), 'I-ADDRESS': (0, 255, 0),
            'B-TIMESTAMP': (255, 0, 0), 'I-TIMESTAMP': (255, 0, 0),
            'B-TOTAL_COST': (0, 255, 255), 'I-TOTAL_COST': (0, 255, 255),
            'B-ITEM': (255, 0, 255), 'I-ITEM': (255, 0, 255),
            'B-QTY': (255, 255, 0), 'I-QTY': (255, 255, 0),
            'B-PRICE': (0, 165, 255), 'I-PRICE': (0, 165, 255),
            'O': (200, 200, 200)
        }
        
        for box, word, label in zip(boxes, words, labels_main):
            if label == 'O':
                continue
            
            box_np = np.array(box, dtype=np.int32)
            color = colors.get(label, (0, 255, 0))
            cv2.polylines(vis_img, [box_np], isClosed=True, color=color, thickness=2)
            
            pt1 = tuple(box_np[0])
            cv2.putText(vis_img, label, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        cv2.imwrite(output_path, vis_img)






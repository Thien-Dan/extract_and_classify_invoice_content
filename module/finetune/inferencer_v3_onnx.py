import os
import torch
import numpy as np
import onnxruntime as ort
from transformers import LayoutLMv3Processor
from PIL import Image

class LayoutLMv3Inferencer:
    def __init__(self, model_dir="saved_models/layoutlmv3_onnx"):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_dir}")
            
        print(f"Loading LayoutLMv3 ONNX model from '{model_dir}'...")
        self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
        self.session = ort.InferenceSession(f"{model_dir}/model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Mapping nhãn cứng vì không dùng model.config nữa
        self.id2label = {
            0: "O", 1: "B-SELLER", 2: "I-SELLER", 3: "B-ADDRESS", 4: "I-ADDRESS", 
            5: "B-TIMESTAMP", 6: "I-TIMESTAMP", 7: "B-TOTAL_COST", 8: "I-TOTAL_COST"
        }
        
    def predict(self, image: Image.Image, words: list, bboxes: list):
        if not words or not bboxes:
            return []
            
        # Lấy encoding trực tiếp dạng numpy
        encoding = self.processor(
            images=image.convert("RGB"),
            text=words,
            boxes=bboxes,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        
        # Word ids phải dùng return_tensors='pt' mới lấy được .word_ids() trên BatchEncoding, 
        # nên ta phải gọi lại hoặc dùng pt roi convert numpy
        encoding_pt = self.processor(
            images=image.convert("RGB"),
            text=words,
            boxes=bboxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        word_ids = encoding_pt.word_ids(batch_index=0)
        
        inputs = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
            "bbox": encoding["bbox"].astype(np.int64),
            "pixel_values": encoding["pixel_values"].astype(np.float32)
        }
        
        outputs = self.session.run(None, inputs)
        logits = outputs[0] # ONNX trả về tuple
        
        # Lấy nhãn dự đoán
        predictions = np.argmax(logits, axis=2).squeeze().tolist()
        
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        predicted_labels = []
        current_word_idx = -1
        
        for pred_id, word_idx in zip(predictions, word_ids):
            if word_idx is None:
                continue
            if word_idx != current_word_idx:
                pred_label = self.id2label.get(pred_id, "O")
                predicted_labels.append(pred_label)
                current_word_idx = word_idx
                
        # Trả về kết quả khớp với số lượng từ gốc
        result_labels = ["O"] * len(words)
        unique_word_ids = list(dict.fromkeys([w for w in word_ids if w is not None]))
        
        for i, word_idx in enumerate(unique_word_ids):
            if i < len(predicted_labels):
                result_labels[word_idx] = predicted_labels[i]
                
        return result_labels


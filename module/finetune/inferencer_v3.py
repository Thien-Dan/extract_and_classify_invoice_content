import os
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

class LayoutLMv3Inferencer:
    def __init__(self, model_dir="saved_models/layoutlmv3_mcocr_4labels"):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_dir}. Hãy chắc chắn đã chạy train_v3.py trước.")
            
        print(f"Loading LayoutLMv3 model from '{model_dir}'...")
        self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        
    def predict(self, image: Image.Image, words: list, bboxes: list):
        if not words or not bboxes:
            return []
            
        encoding = self.processor(
            images=image.convert("RGB"),
            text=words,
            boxes=bboxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        
        # Lấy word_ids TỪ ĐỐI TƯỢNG BatchEncoding TRƯỚC KHI ÉP VỀ DICT!
        word_ids = encoding.word_ids(batch_index=0)
        
        # Chuyển lên thiết bị (GPU/CPU)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        predicted_labels = []
        current_word_idx = -1
        
        for pred_id, word_idx in zip(predictions, word_ids):
            if word_idx is None:
                continue
            if word_idx != current_word_idx:
                pred_label = self.id2label[pred_id]
                predicted_labels.append(pred_label)
                current_word_idx = word_idx
                
        # Trả về kết quả khớp với số lượng từ gốc
        result_labels = ["O"] * len(words)
        unique_word_ids = list(dict.fromkeys([w for w in word_ids if w is not None]))
        
        for i, word_idx in enumerate(unique_word_ids):
            if i < len(predicted_labels):
                result_labels[word_idx] = predicted_labels[i]
                
        return result_labels

import os
import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification

class LayoutLMInferencer:
    def __init__(self, model_dir="saved_models/layoutlm_receipt"):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_dir}. Hãy chắc chắn đã chạy train.py trước.")
            
        print(f"Loading LayoutLM model from '{model_dir}'...")
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_dir)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label
        
    def predict(self, words, bboxes):
        """
        Nhận vào danh sách words và bboxes, trả về danh sách các nhãn dự đoán cho từng word.
        words: List[str]
        bboxes: List[[x0, y0, x1, y1]] (scale 1000)
        
        Returns:
            List[str]: Danh sách các nhãn dự đoán.
        """
        input_ids = [self.tokenizer.cls_token_id]
        final_bboxes = [[0, 0, 0, 0]]
        word_ids = []
        
        for idx, (w, b) in enumerate(zip(words, bboxes)):
            tokens = self.tokenizer.tokenize(w)
            if not tokens:
                continue
                
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids.extend(token_ids)
            final_bboxes.extend([b] * len(token_ids))
            
            word_ids.extend([idx] * len(token_ids))
            
        input_ids.append(self.tokenizer.sep_token_id)
        final_bboxes.append([1000, 1000, 1000, 1000])
        
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)
        bbox_tensor = torch.tensor([final_bboxes], dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids_tensor)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                bbox=bbox_tensor
            )
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        
        # Cẩn thận nếu output chỉ có 1 token (hiếm) thì predictions là int thay vì list
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        current_word_idx = -1
        predicted_labels = []
        
        # Duyệt qua các dự đoán (bỏ token [CLS] đầu và [SEP] cuối)
        for pred_id, word_idx in zip(predictions[1:-1], word_ids):
            if word_idx != current_word_idx:
                pred_label = self.id2label[pred_id]
                predicted_labels.append(pred_label)
                current_word_idx = word_idx
                
        # Để đề phòng lỗi tokenizer nuốt mất word (nếu word = space), ta map lại đúng kích thước
        # Nhưng thông thường length(predicted_labels) <= length(words).
        # Cách chuẩn xác nhất là tạo mảng kết quả rỗng mang nhãn "O"
        result_labels = ["O"] * len(words)
        
        # map lại
        unique_word_ids = list(dict.fromkeys(word_ids))
        for i, word_idx in enumerate(unique_word_ids):
            result_labels[word_idx] = predicted_labels[i]
            
        return result_labels

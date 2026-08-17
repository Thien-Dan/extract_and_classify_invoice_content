import os
import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer

class ReceiptDataset(Dataset):
    def __init__(self, data_dir, tokenizer: LayoutLMTokenizer, label2id, max_seq_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        
        # Đọc danh sách tất cả các file json
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        words = data['words']
        bboxes = data['bboxes']
        labels = data['labels']
        
        # Các mảng kết quả sau khi thực hiện subword alignment
        tokenized_words = []
        tokenized_bboxes = []
        tokenized_labels = []
        
        # Subword Alignment
        for word, box, label in zip(words, bboxes, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
                
            tokenized_words.extend(word_tokens)
            # Nhân bản bbox cho từng subword
            tokenized_bboxes.extend([box] * len(word_tokens))
            
            # Đối với nhãn: Subword đầu tiên mang nhãn B/I gốc, 
            # Các subword phía sau mang nhãn I (Inside) hoặc O.
            if label.startswith("B-"):
                label_first = label
                label_others = "I-" + label[2:]
            else:
                label_first = label
                label_others = label
                
            tokenized_labels.append(label_first)
            tokenized_labels.extend([label_others] * (len(word_tokens) - 1))
            
        # Cắt bớt nếu dài quá max_seq_length - 2 (chừa chỗ cho [CLS] và [SEP])
        if len(tokenized_words) > self.max_seq_length - 2:
            tokenized_words = tokenized_words[:self.max_seq_length - 2]
            tokenized_bboxes = tokenized_bboxes[:self.max_seq_length - 2]
            tokenized_labels = tokenized_labels[:self.max_seq_length - 2]
            
        # Thêm token đặc biệt [CLS] ở đầu và [SEP] ở cuối
        input_ids = [self.tokenizer.cls_token_id]
        final_bboxes = [[0, 0, 0, 0]]
        final_labels = [self.label2id["O"]]
        
        for w, b, l in zip(tokenized_words, tokenized_bboxes, tokenized_labels):
            input_ids.append(self.tokenizer.convert_tokens_to_ids(w))
            final_bboxes.append(b)
            final_labels.append(self.label2id[l] if l in self.label2id else self.label2id["O"])
            
        input_ids.append(self.tokenizer.sep_token_id)
        final_bboxes.append([1000, 1000, 1000, 1000])
        final_labels.append(self.label2id["O"])
        
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # Padding
        padding_length = self.max_seq_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        token_type_ids.extend([0] * padding_length)
        final_bboxes.extend([[0, 0, 0, 0]] * padding_length)
        final_labels.extend([-100] * padding_length) # -100 để PyTorch CrossEntropyLoss bỏ qua
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "bbox": torch.tensor(final_bboxes, dtype=torch.long),
            "labels": torch.tensor(final_labels, dtype=torch.long)
        }

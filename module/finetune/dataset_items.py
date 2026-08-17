import os
import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer

class ReceiptDatasetItems(Dataset):
    def __init__(self, data_dir, tokenizer: LayoutLMTokenizer, label2id, max_seq_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
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
        
        tokenized_words = []
        tokenized_bboxes = []
        tokenized_labels = []
        
        for word, box, label in zip(words, bboxes, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
                
            tokenized_words.extend(word_tokens)
            tokenized_bboxes.extend([box] * len(word_tokens))
            
            # Map everything except ITEM, QTY, PRICE to 'O'
            if label not in self.label2id:
                label = "O"
            
            if label.startswith("B-"):
                label_first = label
                label_others = "I-" + label[2:]
                # Check if I- is in label2id (it should be)
                if label_others not in self.label2id:
                    label_others = "O"
            else:
                label_first = label
                label_others = label
                
            tokenized_labels.append(label_first)
            tokenized_labels.extend([label_others] * (len(word_tokens) - 1))
            
        if len(tokenized_words) > self.max_seq_length - 2:
            tokenized_words = tokenized_words[:self.max_seq_length - 2]
            tokenized_bboxes = tokenized_bboxes[:self.max_seq_length - 2]
            tokenized_labels = tokenized_labels[:self.max_seq_length - 2]
            
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
        
        padding_length = self.max_seq_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        token_type_ids.extend([0] * padding_length)
        final_bboxes.extend([[0, 0, 0, 0]] * padding_length)
        final_labels.extend([-100] * padding_length) 
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "bbox": torch.tensor(final_bboxes, dtype=torch.long),
            "labels": torch.tensor(final_labels, dtype=torch.long)
        }

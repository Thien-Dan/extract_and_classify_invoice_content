import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor

class ReceiptDatasetV3(Dataset):
    def __init__(self, data_dir, processor: LayoutLMv3Processor, label2id, max_seq_length=512):
        self.json_dir = os.path.join(data_dir, "jsons")
        self.img_dir = os.path.join(data_dir, "images")
        self.processor = processor
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        
        self.files = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_name = self.files[idx]
        base_name = os.path.splitext(file_name)[0]
        json_path = os.path.join(self.json_dir, file_name)
        img_path = os.path.join(self.img_dir, base_name + ".jpg")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        words = data['words']
        bboxes = data['bboxes']
        str_labels = data['labels']
        
        # Convert string labels to IDs
        ner_tags = [self.label2id.get(label, 0) for label in str_labels]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        encoding = self.processor(
            images=image,
            text=words,
            boxes=bboxes,
            word_labels=ner_tags,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bbox': encoding['bbox'].squeeze(),
            'labels': encoding['labels'].squeeze()
        }

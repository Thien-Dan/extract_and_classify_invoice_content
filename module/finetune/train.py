import os
import torch
import numpy as np
from torch.utils.data import random_split
from transformers import (
    LayoutLMTokenizer, 
    LayoutLMForTokenClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate

from module.synthetic_data.labels import LayoutLMLabels
from module.finetune.dataset import ReceiptDataset
from module.finetune.config import LayoutLMTrainingConfig

def main():
    print("=== KHỞI TẠO TIẾN TRÌNH FINE-TUNE LAYOUTLM ===")
    
    config = LayoutLMTrainingConfig()
    
    # 1. Khởi tạo Labels & Dictionary
    labels_list = LayoutLMLabels.get_all_labels()
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(labels_list)
    
    print(f"Tổng số nhãn (Labels): {num_labels}")
    
    # 2. Khởi tạo Tokenizer & Dataset
    print(f"Đang tải Tokenizer từ '{config.MODEL_NAME}'...")
    tokenizer = LayoutLMTokenizer.from_pretrained(config.MODEL_NAME)
    
    print(f"Đang tải dữ liệu từ '{config.DATA_DIR}'...")
    full_dataset = ReceiptDataset(
        data_dir=config.DATA_DIR, 
        tokenizer=tokenizer, 
        label2id=label2id,
        max_seq_length=config.MAX_SEQ_LENGTH
    )
    
    # Split train/val
    total_size = len(full_dataset)
    val_size = int(total_size * config.VAL_SPLIT)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dữ liệu Train: {train_size} mẫu | Validation: {val_size} mẫu")
    
    # 3. Định nghĩa Model
    print("Đang khởi tạo mô hình LayoutLM For Token Classification...")
    model = LayoutLMForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )
    
    # 4. Cấu hình Training
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 5. Hàm tính toán độ chính xác (Metrics)
    metric = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Loại bỏ các token mang nhãn -100 (Padding)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    # 6. Bắt đầu Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\n🚀 Bắt đầu quá trình huấn luyện (Training)...")
    trainer.train()
    
    print(f"✅ Đã train xong! Lưu mô hình xuất sắc nhất tại: {config.OUTPUT_DIR}")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

if __name__ == "__main__":
    main()

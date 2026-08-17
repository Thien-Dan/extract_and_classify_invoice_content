import os
import torch
from torch.utils.data import random_split, DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from module.finetune.config_v3 import LayoutLMv3TrainingConfig
from module.finetune.dataset_v3 import ReceiptDatasetV3
from torch.optim import AdamW
from tqdm import tqdm

def train():
    config = LayoutLMv3TrainingConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    labels = ["O", "B-SELLER", "I-SELLER", "B-ADDRESS", "I-ADDRESS", "B-TIMESTAMP", "I-TIMESTAMP", "B-TOTAL_COST", "I-TOTAL_COST"]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    print("1. Loading Processor...")
    processor = LayoutLMv3Processor.from_pretrained(config.MODEL_NAME, apply_ocr=False)
    
    print("2. Preparing Dataset...")
    full_dataset = ReceiptDatasetV3(
        data_dir=config.DATA_DIR, 
        processor=processor, 
        label2id=label2id,
        max_seq_length=config.MAX_SEQ_LENGTH
    )
    
    total_size = len(full_dataset)
    val_size = int(total_size * config.VAL_SPLIT)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    print(f"Train Data: {train_size} samples | Validation: {val_size} samples")
    
    print("3. Loading Model...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    print("\n Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f" => Updated Best Model (Val Loss: {best_val_loss:.4f})")
            model.save_pretrained(config.OUTPUT_DIR)
            processor.save_pretrained(config.OUTPUT_DIR)

if __name__ == "__main__":
    train()

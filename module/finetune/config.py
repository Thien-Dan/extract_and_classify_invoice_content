import os

class LayoutLMTrainingConfig:
    # Model
    MODEL_NAME = "microsoft/layoutlm-base-uncased"
    
    # Paths
    DATA_DIR = "dataset/synthetic_1000/jsons"
    OUTPUT_DIR = "saved_models/layoutlm_receipt"
    
    # Hyperparameters
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 5
    
    # Khác
    VAL_SPLIT = 0.2

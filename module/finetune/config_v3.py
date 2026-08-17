import os

class LayoutLMv3TrainingConfig:
    # Model
    MODEL_NAME = "microsoft/layoutlmv3-base"
    
    # Paths
    DATA_DIR = "dataset/real_data_v3"
    OUTPUT_DIR = "saved_models/layoutlmv3_mcocr_4labels"
    
    # Hyperparameters
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 4 # LayoutLMv3 is heavy on memory due to images
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    
    # Khác
    VAL_SPLIT = 0.1

class LayoutLMTrainingConfigItems:
    MODEL_NAME = "microsoft/layoutlm-base-uncased"
    DATA_DIR = "dataset/synthetic_3000/jsons"
    OUTPUT_DIR = "saved_models/layoutlm_synthetic_items"
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 4
    VAL_SPLIT = 0.2

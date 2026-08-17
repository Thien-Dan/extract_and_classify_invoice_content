import os
import torch

class Config:
    # 1. Base Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
    
    # 2. VietOCR Configuration
    VIETOCR_WEIGHTS = os.path.join(SAVED_MODELS_DIR, 'vietocr_mcocr', 'weights.pth')
    VIETOCR_CONFIG_NAME = 'vgg_transformer'
    
    # Mô hình LayoutLM 4 nhãn mới
    LAYOUTLM_MODEL_DIR = 'saved_models/layoutlm_mcocr_4labels'
    
    # 4. DBNet Configuration
    # Uses internal default paths inside module/DBnet/finetuned_dbnet/
    
    # 5. Device
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 6. Preprocessing
    SKEW_THRESHOLD = 2.0

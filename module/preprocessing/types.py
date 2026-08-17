from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class CropConfig:
    mask_threshold: Optional[int] = None       # None = Otsu tự động
    morph_kernel_size: int = 15
    min_area_ratio: float = 0.05
    max_area_ratio: float = 0.999               # QUAN TRỌNG: gần 1.0
    approx_epsilon_ratio: float = 0.02
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 6.0
    # Độ nở của khung cắt (phần trăm). Ví dụ: 0.03 = mở rộng 3% mỗi cạnh
    expand_ratio: float = 0.03
    
    # Hệ số phóng to độ phân giải sau khi bẻ thẳng (dùng cv2.INTER_LANCZOS4)
    # Ví dụ: 2.0 = nhân đôi chiều rộng và chiều cao ảnh kết quả
    upscale_factor: float = 1.0
    enable_classic_fallback: bool = True
    debug: bool = False

@dataclass
class CropResult:
    success: bool
    method_used: str          # "saliency" | "classic_fallback" | "none"
    warped_image: Optional[np.ndarray]
    quad_points: Optional[np.ndarray]
    transform_matrix: Optional[np.ndarray] = None
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    message: str = ""

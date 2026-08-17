import numpy as np
import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache session toàn cục để không load lại model mỗi lần gọi hàm
_rembg_session = None

def get_saliency_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Chạy U²-Net (qua rembg, only_mask=True) lấy soft saliency mask (0–255).
    """
    global _rembg_session
    try:
        import rembg
    except ImportError:
        raise ImportError("rembg is not installed. Please install it using `pip install rembg[onnxruntime]`.")

    if _rembg_session is None:
        logger.info("Khởi tạo rembg session (u2net)...")
        # u2net is the default and general model for object saliency
        _rembg_session = rembg.new_session("u2net")
        
    logger.debug("Đang trích xuất mask bằng U²-Net...")
    # [TỐI ƯU HÓA] Thêm viền xám 50px xung quanh ảnh để giả lập "background".
    # Giúp U2-Net nhận diện chính xác kể cả khi hóa đơn chiếm trọn 100% khung hình.
    pad_size = 50
    image_rgb = image_bgr[:, :, ::-1]
    padded_image = cv2.copyMakeBorder(
        image_rgb, pad_size, pad_size, pad_size, pad_size, 
        cv2.BORDER_CONSTANT, value=[128, 128, 128]
    )
    
    # only_mask=True trả về grayscale mask 0-255
    padded_mask = rembg.remove(padded_image, session=_rembg_session, only_mask=True)
    
    # Loại bỏ viền 50px đã thêm vào khỏi mask
    h, w = image_bgr.shape[:2]
    mask = padded_mask[pad_size:pad_size+h, pad_size:pad_size+w]
    
    return mask

import cv2
import numpy as np
import logging
from typing import Optional, Union

from .types import CropConfig, CropResult
from .geometry import is_valid_quad, four_point_transform
from .saliency import get_saliency_mask
from .contour import clean_mask, largest_valid_contour, contour_to_quad
from .classic_fallback import classic_contour_detect

logger = logging.getLogger(__name__)

def detect_and_crop_receipt(
    image_input: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    cfg: Optional[CropConfig] = None,
) -> CropResult:
    """
    Main function to run the Saliency-based receipt cropping pipeline.
    """
    if cfg is None:
        cfg = CropConfig()

    debug_images = {}
    
    try:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return CropResult(
                    success=False,
                    method_used="none",
                    warped_image=None,
                    quad_points=None,
                    message=f"Không thể đọc file ảnh: {image_input}"
                )
        else:
            image = image_input.copy()
            
        h, w = image.shape[:2]
        quad = None
        method = "none"
        msg = ""

        # Bước a: Saliency Mask
        saliency_success = False
        try:
            mask = get_saliency_mask(image)
            if cfg.debug:
                debug_images["1_raw_mask"] = mask
                
            # Bước b & c: Clean mask (Threshold + Morphology)
            cleaned_mask = clean_mask(mask, cfg)
            if cfg.debug:
                debug_images["2_cleaned_mask"] = cleaned_mask
                
            # Bước d: Tìm contour hợp lệ
            c = largest_valid_contour(cleaned_mask, cfg)
            if c is not None:
                # Bước e: Lấy tứ giác
                potential_quad = contour_to_quad(c, cfg)
                if potential_quad is not None:
                    # Bước f: Kiểm tra sanity
                    if is_valid_quad(potential_quad, (h, w), cfg):
                        quad = potential_quad
                        saliency_success = True
                        method = "saliency"
                        msg = "Saliency contour detection successful."
                    else:
                        msg = "Saliency quad found but failed aspect ratio sanity check."
                else:
                    msg = "Saliency contour found but failed to convert to quad."
            else:
                msg = "Saliency mask generated but no valid contour found."
                
        except Exception as e:
            msg = f"Saliency exception: {e}"
            logger.warning(msg)

        # Fallback bắt buộc: nếu saliency thất bại
        if not saliency_success and cfg.enable_classic_fallback:
            logger.info("Chuyển sang Classic Fallback...")
            potential_quad = classic_contour_detect(image, cfg)
            if potential_quad is not None and is_valid_quad(potential_quad, (h, w), cfg):
                quad = potential_quad
                method = "classic_fallback"
                msg = "Classic fallback successful."
            else:
                msg += " | Classic fallback also failed."

        # Nếu hoàn toàn không tìm được quad nào
        if quad is None:
            return CropResult(
                success=False,
                method_used="none",
                warped_image=None,
                quad_points=None,
                debug_images=debug_images,
                message=msg
            )
            
        # Mở rộng (padding) một chút ra ngoài để tránh lẹm chữ sát lề
        from .geometry import expand_quad
        quad = expand_quad(quad, (h, w), cfg.expand_ratio)
        
        # Bước g: Four-point perspective transform (kết hợp upscale phân giải nếu có)
        warped, M = four_point_transform(image, quad, upscale_factor=cfg.upscale_factor)
        
        # Ghi ra file nếu có yêu cầu
        if output_path is not None:
            cv2.imwrite(output_path, warped)

        return CropResult(
            success=True,
            method_used=method,
            warped_image=warped,
            quad_points=quad,
            transform_matrix=M,
            debug_images=debug_images,
            message=msg
        )
        
    except Exception as e:
        return CropResult(
            success=False,
            method_used="none",
            warped_image=None,
            quad_points=None,
            message=f"Fatal error: {str(e)}"
        )

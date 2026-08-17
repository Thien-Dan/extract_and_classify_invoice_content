import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImageDeskewer:
    def __init__(self, skew_threshold=1.5):
        """
        skew_threshold: Góc nghiêng tối thiểu (độ) để thực hiện xoay ảnh.
        """
        self.skew_threshold = skew_threshold

    def calculate_angle_from_boxes(self, boxes):
        """
        Tính toán góc nghiêng trung bình của văn bản dựa trên danh sách các khung chữ (DBNet).
        boxes: numpy array shape (N, 4, 2)
        """
        if len(boxes) == 0:
            return 0.0

        angles = []
        for box in boxes:
            if isinstance(box, list):
                box = np.array(box)
                
            # Sắp xếp 4 điểm: theo x
            # Để tính toán cạnh dài nhất (ngang hay dọc)
            # Dùng cv2.minAreaRect là cách chuẩn nhất
            rect = cv2.minAreaRect(box.astype(np.float32))
            center, size, angle = rect
            w, h = size

            # OpenCV minAreaRect trả về góc:
            # Cũ: [-90, 0)
            # Mới: (0, 90]
            # Ta cần chuyển về góc thực tế của dòng chữ.
            # Cách đơn giản: Tính vector của cạnh dài nhất.
            
            p0, p1, p2, p3 = box
            # Lấy 2 cạnh liên tiếp
            edge1 = p1 - p0
            edge2 = p2 - p1
            
            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)
            
            # Cạnh nào dài hơn thì đó là cạnh chạy dọc theo chiều dài chữ
            if len1 > len2:
                dx, dy = edge1
            else:
                dx, dy = edge2
                
            # Tránh chia cho 0
            if dx == 0:
                dx = 1e-5
                
            # Tính góc bằng arctan2, đổi sang độ
            a = np.arctan2(dy, dx) * 180.0 / np.pi
            
            # Đưa góc về khoảng [-90, 90]
            if a > 90:
                a -= 180
            elif a < -90:
                a += 180
                
            angles.append(a)

        # Trả về góc trung vị để loại bỏ nhiễu
        median_angle = np.median(angles)
        return median_angle

    def deskew_and_rotate(self, image, boxes):
        """
        Phân tích góc xoay từ các boxes và xoay lại ảnh cho thẳng.
        Trả về ảnh đã xoay thẳng.
        """
        angle = self.calculate_angle_from_boxes(boxes)
        
        # Nếu góc nhỏ, ta coi như chỉ là nghiêng (skew)
        # Nếu góc gần 90 hoặc -90, ta coi như bị xoay ngang dọc
        # Xoay theo góc đó.
        
        if abs(angle) < self.skew_threshold:
            # Quá nhỏ, không cần xoay
            logger.info(f"Deskewer: Góc nghiêng quá nhỏ ({angle:.2f} độ), bỏ qua xoay.")
            return image, 0.0
            
        logger.info(f"Deskewer: Phát hiện ảnh nghiêng {angle:.2f} độ. Đang nắn thẳng...")
        
        # Cần xoay lại một góc -angle
        rotation_angle = angle # OpenCV rotation matrix xoay theo chiều kim đồng hồ nếu dương,
                               # Nhưng np.arctan2 trên ảnh y hướng xuống: 
                               # Góc dương nghĩa là cạnh chữ đi xuống (xuôi chiều kim đồng hồ)
                               # Ta cần xoay NGƯỢC lại, nên góc trong getRotationMatrix2D dương (để xoay ngược kim đồng hồ)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Tính toán kích thước ảnh mới sau khi xoay để không bị cắt xén
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        
        # Cập nhật ma trận tịnh tiến
        M[0, 2] += bound_w / 2 - center[0]
        M[1, 2] += bound_h / 2 - center[1]
        
        # Xoay
        rotated = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, rotation_angle

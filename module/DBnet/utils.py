import cv2
import numpy as np

def get_rotated_crop(image, box):
    """
    Cắt và nắn thẳng (Perspective Transform) một vùng ảnh dựa trên tọa độ 4 góc.
    
    Args:
        image (numpy.ndarray): Ảnh gốc (BGR hoặc RGB).
        box (list hoặc numpy.ndarray): Tọa độ 4 góc [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        
    Returns:
        numpy.ndarray: Ảnh đã được cắt và nắn thẳng thành hình chữ nhật.
    """
    box = np.array(box, dtype=np.float32)
    
    # Tính chiều rộng của ảnh mới (khoảng cách lớn nhất giữa góc trên hoặc dưới)
    width = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3])
    ))
    
    # Tính chiều cao của ảnh mới (khoảng cách lớn nhất giữa góc trái hoặc phải)
    height = int(max(
        np.linalg.norm(box[1] - box[2]),
        np.linalg.norm(box[3] - box[0])
    ))
    
    # Tọa độ 4 góc của bức ảnh đích (hình chữ nhật hoàn hảo)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Ma trận biến đổi không gian (Perspective Transform)
    M = cv2.getPerspectiveTransform(box, dst_pts)
    
    # Áp dụng ma trận để nắn và cắt ảnh
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped

def deskew_image(image):
    """
    Tự động phát hiện và xoay thẳng lại (deskew) ảnh hóa đơn bị nghiêng.
    Sử dụng kỹ thuật tìm viền Contour và minAreaRect của OpenCV.
    
    Args:
        image (numpy.ndarray): Ảnh đầu vào (BGR).
        
    Returns:
        tuple: (Ảnh đã được nắn thẳng, Góc nghiêng đã phát hiện)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize ảnh (đen trắng)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Dùng kernel dài theo chiều ngang để kết nối các ký tự thành những khối chữ (block)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    
    # Tìm contours (các khối chữ)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100: # Bỏ qua nhiễu nhỏ
            rect = cv2.minAreaRect(c)
            width, height = rect[1]
            angle = rect[2]
            
            # Cân bằng góc của minAreaRect (quy về khoảng -45 đến 45 độ)
            if width < height:
                angle = angle + 90
            
            # Chỉ lấy các góc nghiêng nhẹ (tránh trường hợp ảnh dọc bị xoay ngang 90 độ)
            if -45 < angle < 45:
                angles.append(angle)
            
    if len(angles) == 0:
        return image, 0.0
        
    angles.sort()
    # Lấy góc trung vị (median) để loại bỏ các góc bị nhiễu do hình vẽ/logo
    median_angle = np.median(angles)
    
    # Nếu ảnh nghiêng quá ít (< 0.5 độ) thì giữ nguyên để tiết kiệm thời gian
    if abs(median_angle) < 0.5:
        return image, median_angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Tạo ma trận xoay. Lưu ý: truyền median_angle vào trực tiếp để undo sự nghiêng
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Xoay ảnh với viền màu trắng để phần khoảng trống không bị đen xì
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated, median_angle

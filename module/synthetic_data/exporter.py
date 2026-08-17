import os
import json
from .generator import ReceiptGenerator

def export_dataset(num_samples=100, output_dir="dataset/synthetic"):
    """
    Sinh và lưu hàng loạt dữ liệu giả vào thư mục.
    
    Args:
        num_samples (int): Số lượng ảnh muốn tạo.
        output_dir (str): Thư mục lưu trữ.
    """
    img_dir = os.path.join(output_dir, "images")
    json_dir = os.path.join(output_dir, "jsons")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
        
    print(f"Bắt đầu sinh {num_samples} ảnh hóa đơn giả lập...")
    generator = ReceiptGenerator()
    
    for i in range(num_samples):
        # Format index, e.g., 0001, 0002
        idx_str = f"{i:04d}"
        
        # Gọi hàm sinh
        image, layout_data = generator.generate_receipt()
        
        # Tên file
        img_filename = f"receipt_{idx_str}.jpg"
        json_filename = f"receipt_{idx_str}.json"
        
        img_path = os.path.join(img_dir, img_filename)
        json_path = os.path.join(json_dir, json_filename)
        
        # Lưu ảnh
        image.save(img_path, "JPEG")
        
        # Lưu JSON annotation
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=4)
            
        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f">> Đã sinh được {i + 1} / {num_samples} mẫu.")
            
    print(f"Hoàn tất! Dữ liệu đã được lưu tại thư mục: {output_dir} (với 2 thư mục con images/ và jsons/)")
